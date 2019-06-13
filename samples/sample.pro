include($${PWD}/../common.pri)

TEMPLATE = app
TARGET = sample.out

BUILD_DIR = build
DESTDIR = $${BUILD_DIR}/bin
OBJECTS_DIR = $${BUILD_DIR}/obj
CUDA_OBJECTS_DIR = $${BUILD_DIR}/cudaobj

INCLUDEPATH += $${PWD}/../split/include

CUDA_SOURCES += $$files(src/*.cu, true) 

SPLIT_LIB_PATH = $${PWD}/../split/build/lib
LIBS += -L$${SPLIT_LIB_PATH} -lsplit 

include($${PWD}/../cuda_compiler.pri)

QMAKE_RPATHDIR += $${SPLIT_LIB_PATH}

