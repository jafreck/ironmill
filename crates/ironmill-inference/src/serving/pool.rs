//! vAttention-style contiguous KV cache memory pool.
//!
//! [`KvPool`] is a CPU-side allocator that tracks byte offsets into a
//! virtual backing buffer. The actual Metal/GPU buffer operations happen
//! in the inference engine implementation — this module only does
//! bookkeeping.

use std::collections::BTreeMap;

use crate::engine::{InferenceError, SequenceId};

/// A contiguous sub-allocation within the KV cache pool.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct KvAllocation {
    /// Byte offset into the backing buffer.
    pub offset: usize,
    /// Total allocated capacity (in slots).
    pub capacity: usize,
    /// Number of slots currently in use.
    pub used: usize,
}

/// vAttention-style contiguous KV cache memory pool.
///
/// Manages a fixed-size virtual backing buffer using a first-fit free
/// list allocator. Supports allocation, deallocation, in-place growth
/// (with fallback relocation), and defragmentation.
pub struct KvPool {
    backing_size: usize,
    allocations: BTreeMap<SequenceId, KvAllocation>,
    free_list: Vec<(usize, usize)>, // (offset, length), kept sorted by offset
}

impl KvPool {
    /// Create a new pool with the given total backing size.
    pub fn new(backing_size: usize) -> Self {
        Self {
            backing_size,
            allocations: BTreeMap::new(),
            free_list: vec![(0, backing_size)],
        }
    }

    /// Allocate a contiguous region for the given sequence.
    ///
    /// Uses first-fit strategy over the sorted free list.
    pub fn allocate(
        &mut self,
        id: SequenceId,
        initial_capacity: usize,
    ) -> Result<&KvAllocation, InferenceError> {
        if initial_capacity == 0 {
            return Err(InferenceError::Allocation(
                "initial_capacity must be > 0".into(),
            ));
        }
        if self.allocations.contains_key(&id) {
            return Err(InferenceError::Allocation(format!(
                "sequence {id} already has an allocation"
            )));
        }

        let region_idx = self
            .free_list
            .iter()
            .position(|&(_, len)| len >= initial_capacity)
            .ok_or_else(|| {
                InferenceError::Allocation(format!(
                    "no free region of size {initial_capacity} (total free: {})",
                    self.free_space()
                ))
            })?;

        let (offset, length) = self.free_list[region_idx];
        if length == initial_capacity {
            self.free_list.remove(region_idx);
        } else {
            self.free_list[region_idx] = (offset + initial_capacity, length - initial_capacity);
        }

        self.allocations.insert(
            id,
            KvAllocation {
                offset,
                capacity: initial_capacity,
                used: 0,
            },
        );

        Ok(&self.allocations[&id])
    }

    /// Free the allocation for the given sequence, returning its memory
    /// to the free list.
    pub fn free(&mut self, id: SequenceId) -> Result<(), InferenceError> {
        let alloc = self
            .allocations
            .remove(&id)
            .ok_or(InferenceError::SequenceNotFound(id))?;

        self.free_list.push((alloc.offset, alloc.capacity));
        self.coalesce_free_list();
        Ok(())
    }

    /// Grow the allocation for the given sequence.
    ///
    /// Tries to extend in-place first (adjacent free space). Falls back
    /// to relocating the allocation to a larger free region.
    pub fn grow(&mut self, id: SequenceId) -> Result<(), InferenceError> {
        let alloc = self
            .allocations
            .get(&id)
            .ok_or(InferenceError::SequenceNotFound(id))?;

        let old_end = alloc.offset + alloc.capacity;
        let grow_by = alloc.capacity.max(1);

        // Try in-place extension into adjacent free space.
        if let Some(idx) = self
            .free_list
            .iter()
            .position(|&(off, len)| off == old_end && len >= grow_by)
        {
            let (free_off, free_len) = self.free_list[idx];
            if free_len == grow_by {
                self.free_list.remove(idx);
            } else {
                self.free_list[idx] = (free_off + grow_by, free_len - grow_by);
            }
            let alloc = self.allocations.get_mut(&id).unwrap();
            alloc.capacity += grow_by;
            return Ok(());
        }

        // Relocate: find a free region for the expanded size.
        let new_capacity = alloc.capacity + grow_by;
        let old_offset = alloc.offset;
        let old_capacity = alloc.capacity;
        let old_used = alloc.used;

        let region_idx = self
            .free_list
            .iter()
            .position(|&(_, len)| len >= new_capacity)
            .ok_or_else(|| {
                InferenceError::Allocation(format!(
                    "cannot grow sequence {id}: no free region of size {new_capacity}"
                ))
            })?;

        let (new_offset, free_len) = self.free_list[region_idx];
        if free_len == new_capacity {
            self.free_list.remove(region_idx);
        } else {
            self.free_list[region_idx] = (new_offset + new_capacity, free_len - new_capacity);
        }

        // Return old region to free list.
        self.free_list.push((old_offset, old_capacity));
        self.coalesce_free_list();

        let alloc = self.allocations.get_mut(&id).unwrap();
        alloc.offset = new_offset;
        alloc.capacity = new_capacity;
        alloc.used = old_used;

        Ok(())
    }

    /// Defragment the pool by compacting all allocations toward offset 0.
    ///
    /// Returns a list of relocations as `(sequence_id, old_offset, new_offset)`.
    /// The caller (inference engine) is responsible for copying the
    /// actual KV cache data accordingly.
    pub fn defragment(&mut self) -> Result<Vec<(SequenceId, usize, usize)>, InferenceError> {
        // Sort allocations by current offset.
        let mut sorted: Vec<(SequenceId, usize, usize)> = self
            .allocations
            .iter()
            .map(|(&id, a)| (id, a.offset, a.capacity))
            .collect();
        sorted.sort_by_key(|&(_, offset, _)| offset);

        let mut relocations = Vec::new();
        let mut cursor: usize = 0;

        for (id, old_offset, capacity) in sorted {
            if old_offset != cursor {
                relocations.push((id, old_offset, cursor));
                let alloc = self.allocations.get_mut(&id).unwrap();
                alloc.offset = cursor;
            }
            cursor += capacity;
        }

        // Rebuild free list: single contiguous region at the end.
        self.free_list.clear();
        if cursor < self.backing_size {
            self.free_list.push((cursor, self.backing_size - cursor));
        }

        Ok(relocations)
    }

    /// Look up an allocation by sequence ID.
    pub fn get(&self, id: SequenceId) -> Option<&KvAllocation> {
        self.allocations.get(&id)
    }

    /// Mutably look up an allocation by sequence ID.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut KvAllocation> {
        self.allocations.get_mut(&id)
    }

    /// Total free space across all free regions.
    pub fn free_space(&self) -> usize {
        self.free_list.iter().map(|&(_, len)| len).sum()
    }

    /// Number of active allocations.
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Merge adjacent free regions in the free list.
    fn coalesce_free_list(&mut self) {
        self.free_list.sort_by_key(|&(offset, _)| offset);
        let mut i = 0;
        while i + 1 < self.free_list.len() {
            let (off_a, len_a) = self.free_list[i];
            let (off_b, len_b) = self.free_list[i + 1];
            if off_a + len_a == off_b {
                self.free_list[i] = (off_a, len_a + len_b);
                self.free_list.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_pool_allocate_basic() {
        let mut pool = KvPool::new(1024);
        let alloc = pool.allocate(1, 128).unwrap();
        assert_eq!(alloc.offset, 0);
        assert_eq!(alloc.capacity, 128);
        assert_eq!(alloc.used, 0);
        assert_eq!(pool.free_space(), 1024 - 128);
    }

    #[test]
    fn serving_pool_allocate_multiple() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 256).unwrap();
        let alloc2 = pool.allocate(2, 256).unwrap();
        assert_eq!(alloc2.offset, 256);
        assert_eq!(pool.free_space(), 512);
        assert_eq!(pool.allocation_count(), 2);
    }

    #[test]
    fn serving_pool_allocate_duplicate_id_fails() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 128).unwrap();
        assert!(pool.allocate(1, 128).is_err());
    }

    #[test]
    fn serving_pool_allocate_zero_capacity_fails() {
        let mut pool = KvPool::new(1024);
        assert!(pool.allocate(1, 0).is_err());
    }

    #[test]
    fn serving_pool_allocate_too_large_fails() {
        let mut pool = KvPool::new(256);
        assert!(pool.allocate(1, 512).is_err());
    }

    #[test]
    fn serving_pool_free_reclaims_memory() {
        let mut pool = KvPool::new(512);
        pool.allocate(1, 256).unwrap();
        pool.allocate(2, 256).unwrap();
        assert_eq!(pool.free_space(), 0);

        pool.free(1).unwrap();
        assert_eq!(pool.free_space(), 256);

        // Can re-use the freed space.
        let alloc = pool.allocate(3, 128).unwrap();
        assert_eq!(alloc.offset, 0);
        assert_eq!(pool.free_space(), 128);
    }

    #[test]
    fn serving_pool_free_unknown_id_fails() {
        let mut pool = KvPool::new(1024);
        assert!(pool.free(999).is_err());
    }

    #[test]
    fn serving_pool_free_coalesces_adjacent() {
        let mut pool = KvPool::new(512);
        pool.allocate(1, 128).unwrap(); // [0..128)
        pool.allocate(2, 128).unwrap(); // [128..256)
        pool.allocate(3, 128).unwrap(); // [256..384)

        // Free the middle, then the first — should coalesce into one region.
        pool.free(2).unwrap();
        pool.free(1).unwrap();
        assert_eq!(pool.free_space(), 384); // 512 - 128 (only seq 3 remains)

        // Should be able to allocate a single 256-slot region at offset 0.
        let alloc = pool.allocate(4, 256).unwrap();
        assert_eq!(alloc.offset, 0);
        assert_eq!(alloc.capacity, 256);
    }

    #[test]
    fn serving_pool_grow_in_place() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 128).unwrap();
        assert_eq!(pool.get(1).unwrap().capacity, 128);

        pool.grow(1).unwrap();
        let alloc = pool.get(1).unwrap();
        assert_eq!(alloc.capacity, 256); // doubled
        assert_eq!(alloc.offset, 0); // didn't move
    }

    #[test]
    fn serving_pool_grow_with_relocation() {
        let mut pool = KvPool::new(512);
        pool.allocate(1, 128).unwrap(); // [0..128)
        pool.allocate(2, 128).unwrap(); // [128..256)

        // Growing seq 1 can't extend in-place (seq 2 is adjacent).
        // It should relocate to [256..512).
        pool.grow(1).unwrap();
        let alloc = pool.get(1).unwrap();
        assert_eq!(alloc.capacity, 256);
        assert_eq!(alloc.offset, 256);
    }

    #[test]
    fn serving_pool_grow_unknown_id_fails() {
        let mut pool = KvPool::new(1024);
        assert!(pool.grow(999).is_err());
    }

    #[test]
    fn serving_pool_defragment_compacts() {
        let mut pool = KvPool::new(1024);
        pool.allocate(1, 128).unwrap(); // [0..128)
        pool.allocate(2, 128).unwrap(); // [128..256)
        pool.allocate(3, 128).unwrap(); // [256..384)

        // Free the middle sequence — creates a gap.
        pool.free(2).unwrap();

        let relocations = pool.defragment().unwrap();
        // Sequence 3 should move from 256 → 128.
        assert_eq!(relocations.len(), 1);
        assert_eq!(relocations[0], (3, 256, 128));

        // Pool should now have a single contiguous free region.
        assert_eq!(pool.free_space(), 1024 - 256);
        let alloc3 = pool.get(3).unwrap();
        assert_eq!(alloc3.offset, 128);
    }

    #[test]
    fn serving_pool_defragment_no_gaps() {
        let mut pool = KvPool::new(512);
        pool.allocate(1, 128).unwrap();
        pool.allocate(2, 128).unwrap();

        let relocations = pool.defragment().unwrap();
        assert!(relocations.is_empty());
    }

    #[test]
    fn serving_pool_four_concurrent_sequences() {
        let mut pool = KvPool::new(1024);

        for id in 1..=4u64 {
            let alloc = pool.allocate(id, 128).unwrap();
            assert_eq!(alloc.offset, (id as usize - 1) * 128);
        }
        assert_eq!(pool.allocation_count(), 4);
        assert_eq!(pool.free_space(), 1024 - 512);

        // Free two and verify reclamation.
        pool.free(2).unwrap();
        pool.free(4).unwrap();
        assert_eq!(pool.allocation_count(), 2);
        assert_eq!(pool.free_space(), 1024 - 256);

        // Defragment and verify compaction.
        let relocations = pool.defragment().unwrap();
        assert!(relocations.len() <= 2);
        assert_eq!(pool.free_space(), 1024 - 256);
    }

    #[test]
    fn serving_pool_full_lifecycle() {
        let mut pool = KvPool::new(256);

        // Fill the pool.
        pool.allocate(1, 64).unwrap();
        pool.allocate(2, 64).unwrap();
        pool.allocate(3, 64).unwrap();
        pool.allocate(4, 64).unwrap();
        assert_eq!(pool.free_space(), 0);

        // Can't allocate more.
        assert!(pool.allocate(5, 64).is_err());

        // Free and reallocate.
        pool.free(3).unwrap();
        assert_eq!(pool.free_space(), 64);
        let alloc = pool.allocate(5, 64).unwrap();
        assert_eq!(alloc.offset, 128); // reuses seq 3's slot
    }
}
