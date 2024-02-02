/**
  Based on https://stackoverflow.com/questions/39044063/pass-a-closure-from-cython-to-c
 */

#include <Python.h>
#include <fplll/fplll.h>

extern "C++" {
  bool evaluator_callback_call_obj(PyObject *obj, int n, double *new_sol_coord);
}

class PyCallbackEvaluatorWrapper {
public:
  // constructors and destructors mostly do reference counting
  PyCallbackEvaluatorWrapper(PyObject *o) : held(o)
  {
    Py_XINCREF(o);
  }

  PyCallbackEvaluatorWrapper(const PyCallbackEvaluatorWrapper &rhs)
      : PyCallbackEvaluatorWrapper(rhs.held)
  {
  }

  PyCallbackEvaluatorWrapper(PyCallbackEvaluatorWrapper &&rhs) : held(rhs.held)
  {
    rhs.held = 0;
  }

  // need no-arg constructor to stack allocate in Cython
  PyCallbackEvaluatorWrapper() : PyCallbackEvaluatorWrapper(nullptr)
  {

  }
  ~PyCallbackEvaluatorWrapper()
  {
    Py_XDECREF(held);
  }

  PyCallbackEvaluatorWrapper &operator=(const PyCallbackEvaluatorWrapper &rhs)
  {
    PyCallbackEvaluatorWrapper tmp = rhs;
    return (*this = std::move(tmp));
  }

  PyCallbackEvaluatorWrapper &operator=(PyCallbackEvaluatorWrapper &&rhs)
  {
    held = rhs.held;
    rhs.held = 0;
    return *this;
  }

  bool operator()(size_t n, fplll::enumf *new_sol_coord, void *ctx)
  {
    if (held) // nullptr check
      {
        // note, no way of checking for errors until you return to Python
        return evaluator_callback_call_obj(held, n, new_sol_coord);
      }
    return false;
  }

private:
  PyObject *held;
};
