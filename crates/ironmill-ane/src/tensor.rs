//! IOSurface-backed tensor management for ANE I/O.
//!
//! All ANE inputs and outputs use IOSurface memory. This module manages
//! tensor creation, data transfer, and lifecycle with correct padding
//! and alignment per ANE constraints.

// TODO: Implement AneTensor — see spec Task 3
