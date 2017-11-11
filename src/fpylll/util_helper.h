#pragma once

#include <fplll/enum/enumerate_ext.h>

inline std::function<extenum_fc_enumerate> void_ptr_to_function(void *ptr) {
  return reinterpret_cast<extenum_fc_enumerate*>(ptr);
}
