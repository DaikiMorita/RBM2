#include <Python.h>

// long long 型：64 bit 以上の整数を表現できる
long long fibonacci(unsigned int n){
    if (n<2){
        return 1;
    } else {
        return fibonacci(n-2) + fibonacci(n-1);
    }
}

static  PyObject* fibonacci_py(PyObject* self, PyObject* args){
    PyObject *result = NULL;
    long n;
    long long fib;

    if (PyArg_ParseTuple(args, "l", &n)){
        if (n < 0){
            PyErr_SetString(PyExc_ValueError,"n must note be less than 0");
        } else {
        result = Py_BuildValue("L", fibonacci((unsigned int)n));
        }
    }
    return result;
} 

static char fibonacci_docs[] = "fibonaccci(n): Return nth Fibonacci sequence number" "computed recursively\n";

static PyMethodDef fibonacci_module_methods[] = {
    {"fibonacci", (PyCFunction)fibonacci_py,METH_VARARGS,fibonacci_docs},
    {NULL,NULL,0,NULL}
};

static struct PyModuleDef fibonacci_module_definition = {
    PyModuleDef_HEAD_INIT,
    "fibonacci",
    "Extension module that provides fibonacci sqquence function",
    -1,
    fibonacci_module_methods
};

PyMODINIT_FUNC PyInit_fibonacci(void){
    Py_Initialize();

    return PyModule_Create(&fibonacci_module_definition);
}