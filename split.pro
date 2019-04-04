include($${PWD}/common.pri)

TEMPLATE = app
TARGET = split

BUILD_DIR = build
DESTDIR = $${BUILD_DIR}/bin
OBJECTS_DIR = $${BUILD_DIR}/obj
CUDA_OBJECTS_DIR = $${BUILD_DIR}/cudaobj

INCLUDEPATH += $${PWD}/include

HEADERS += $$files(include/*(.h | .hpp | .inl | cuh), true)
SOURCES += $$files(src/*.cpp, true)
CUDA_SOURCES += $$files(src/*.cu, true) 

include($${PWD}/cuda_compiler.pri)

