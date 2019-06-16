include($${PWD}/../common.pri)

TEMPLATE = lib
TARGET = split

BUILD_DIR = build
DESTDIR = $${BUILD_DIR}/lib
OBJECTS_DIR = $${BUILD_DIR}/obj
CUDA_OBJECTS_DIR = $${BUILD_DIR}/cudaobj

INCLUDEPATH += $${PWD}/include

HEADERS += $$files(include/*(.h | .hpp | .inl | cuh), true)
SOURCES += $$files(src/*.cpp, true)
CUDA_SOURCES += $$files(src/*.cu, true) 

NVCCFLAGS += --shared --cudart=static
include($${PWD}/../cuda_compiler.pri)

