### define COMPILE_MODE to one of these 3 options :
### COMPILE_MODE := debug
### COMPILE_MODE := release
### COMPILE_MODE := demo

EXEEXT=

COMPILE_MODE := debug

UNAME_SHORT := $(shell uname -s)

ifeq ($(UNAME_SHORT),Linux)
  CFLAGS += -DOS_LINUX=1
  DEFINES += -DLINUX -D_REENTRANT -D_GNU_SOURCE
  CFLAGS += -isystem/usr/local/include/opencv
  LDFLAGS += `pkg-config --libs opencv`
  LDFLAGS += -L/usr/local/install/opencv_svn/latest_tested_snapshot/opencv/build/lib/
  LDFLAGS += -lpthread

  CFLAGS += `pkg-config --cflags opencv`
  CFLAGS += -Wall -Wno-system-headers
  CFLAGS += -Werror

  ifeq ($(COMPILE_MODE),debug)
    CFLAGS += -ggdb -O0 -fstack-protector-all
  endif

  ifeq ($(COMPILE_MODE),release)
    CFLAGS += -O99 -ftree-vectorize
    CFLAGS += -ggdb
  endif

## New. Matt's code
GD_LIBS=-lgd -lpng -lz -ljpeg -lfreetype -lm
CDEF_LIBS=-lod3 -lcdef

VPATH=../../../common
INCS=-I${VPATH}

# MACH=-m32
MACH=

LIBDIR=../../john_robinson_3/package/lib64
CFLAGS += -I../../john_robinson_3/package

export LD_LIBRARY_PATH=.:${LIBDIR}

## End new.
endif

ifeq ($(OSTYPE),darwin)
  CFLAGS += -DOS_DARWIN=1

 # CFLAGS += -isystem/usr/local/include/opencv
 # LDFLAGS += `pkg-config --libs opencv`
 # LDFLAGS += -L/Users/user/dev/cmake_binary_dir/lib
	LDFLAGS += -L/Users/user/dev/OpenCV-2.0.0/build_i386/src
	LDFLAGS += -lpthread

##  CFLAGS += `pkg-config --cflags opencv`
  CFLAGS += -Wall -Wno-system-headers
  CFLAGS += -Werror

  ifeq ($(COMPILE_MODE),debug)
    CFLAGS += -ggdb -O0 -fstack-protector-all
  endif

  ifeq ($(COMPILE_MODE),release)
    CFLAGS += -O99 -ftree-vectorize
    CFLAGS += -ggdb
  endif

endif

ifeq ($(UNAME_SHORT),Darwin)
	CFLAGS += -DOS_DARWIN=1
	#CFLAGS += -Isystem/usr/local/include/opencv
	#LDFLAGS += `pkg-config --libs opencv`
	#LDFLAGS += -L/Users/user/dev/OpenCV-2.0.0/build_i386/src
	LDFLAGS += -L/Users/user/dev/cmake_binary_dir/lib
	LDFLAGS += -lpthread
	LDFLAGS +=  -lcv  -l_highgui -l_ml -l_cv

##  CFLAGS += `pkg-config --cflags opencv`
  CFLAGS += -Wall -Wno-system-headers
  CFLAGS += -Werror

  ifeq ($(COMPILE_MODE),debug)
    CFLAGS += -ggdb -O0 -fstack-protector-all
  endif

  ifeq ($(COMPILE_MODE),release)
    CFLAGS += -O99 -ftree-vectorize
    CFLAGS += -ggdb
  endif

endif

ifeq ($(UNAME_SHORT),CYGWIN_NT-5.1)
  OS = win32
endif

ifeq ($(UNAME_SHORT),CYGWIN_NT-6.0)
  OS = win32
endif

ifeq ($(OS),win32)
  EXEEXT=.exe
  CFLAGS += -DWIN32=1 -DWINDOWS=1 -DOS_WINDOWS=1
  CFLAGS += -I"C:\Program Files\OpenCV\cxcore\include"
  CFLAGS += -I"C:\Program Files\OpenCV\cv\include"
  CFLAGS += -I"C:\Program Files\OpenCV\cvaux\include"
  CFLAGS += -I"C:\Program Files\OpenCV\otherlibs\highgui"
  CFLAGS += -I"C:\Program Files\OpenCV\otherlibs\cvcam\include"
  LDFLAGS += -L"C:\Program Files\OpenCV\bin" 
  LDFLAGS += -lcxcore110 -lcv110 -lhighgui110 -lcvaux110
  LDFLAGS += -lpthread


  ifeq ($(COMPILE_MODE),debug)
    CFLAGS += -ggdb -O2
  endif

  ifeq ($(COMPILE_MODE),release)
    CFLAGS += -O99
    CFLAGS += -ggdb
  endif
endif

#CFLAGS += -I../../common

#CFLAGS += ${DEFINES}

CFLAGS += -DNOT_MAC_APP

# CFLAGS += -Winline
# CFLAGS += -fopenmp

# LDFLAGS +=  $(SDLLIBS) -lSDL_ttf -L/usr/X11R6/lib -lX11
# LDFLAGS += -lssp

# SDLLIBS = $(shell sdl-config --libs)
# SDLFLAGS = $(shell sdl-config --cflags | sed "s=-I=-isystem=g")
# CFLAGS += $(SDLFLAGS)



				
#H_FILES = Makefile face_draw.h face_io.h face_results.h cropped_frames.h face_calc.h	
H_FILES =  config.h face_common.h  face_util.h face_draw.h face_io.h face_results.h face_calc.h face_csv.h cropped_frames.h core_common.h core_opencv.h 

all: peter_framing_filter 

clean:
	rm -f makehist *.o core


peter_framing_filter:	Makefile  face_draw.o face_io.o face_results.o face_calc.o cropped_frames.o face_tracker_adjustable_frame.o
	ln -sf libcdef.so.0.0.2 ${LIBDIR}/libcdef.so
	ln -sf libcdef.so.0.0.2 ${LIBDIR}/libcdef.so.0
	ln -sf libod3.so.1.0.2 ${LIBDIR}/libod3.so
	ln -sf libod3.so.1.0.2 ${LIBDIR}/libod3.so.1
	g++ ${CFLAGS} csv.o core_common.o core_opencv.o face_util.o face_draw.o face_io.o face_results.o face_calc.o cropped_frames.o face_tracker_adjustable_frame.o ${LDFLAGS} -L. -L${LIBDIR} ${CDEF_LIBS} -o peter_framing_filter${EXEEXT}

csv.o: ${H_FILES} csv.cpp
	g++ ${CFLAGS} -c csv.cpp
	
core_common.o : ${H_FILES} core_common.cpp
	g++ ${CFLAGS} -c core_common.cpp
	
core_opencv.cpp: ${H_FILES} core_opencv.cpp
	g++ ${CFLAGS} -c core_opencv.cpp			

face_util.cpp: ${H_FILES} face_util.cpp
	g++ ${CFLAGS} -c face_util.cpp
	
face_draw.o: ${H_FILES} face_draw.cpp
	g++ ${CFLAGS} -c face_draw.cpp

face_io.o: ${H_FILES} face_io.cpp
	g++ ${CFLAGS} -c face_io.cpp
	
face_results.o: ${H_FILES} face_results.cpp
	g++ ${CFLAGS} -c face_results.cpp	

face_calc.o: ${H_FILES} face_calc.cpp
	g++ ${CFLAGS} -c face_calc.cpp
	
cropped_frames.o: ${H_FILES} cropped_frames.cpp
	g++ ${CFLAGS} -c cropped_frames.cpp

face_tracker_adjustable_frame.o: ${H_FILES} face_tracker_adjustable_frame.cpp
	g++ ${CFLAGS} -c face_tracker_adjustable_frame.cpp

