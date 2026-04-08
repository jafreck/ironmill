// Exception-safe CFRelease wrapper for Rust Drop implementations.
//
// ObjC exceptions thrown during -dealloc (e.g. when releasing ANE daemon
// connections on machines without ANE hardware) would otherwise propagate
// through Rust frames, causing "Rust cannot catch foreign exceptions" aborts.

#import <CoreFoundation/CoreFoundation.h>

void ane_safe_cfrelease(const void *cf) {
    if (cf == NULL) return;
    @try {
        CFRelease(cf);
    } @catch (id exception) {
        // Swallow — nothing useful we can do in a destructor.
    }
}
