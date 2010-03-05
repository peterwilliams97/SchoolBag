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
 # LDFLAGS += -L/usr/local/install/opencv_svn/latest_tested_snapshot/opencv/build/lib/
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


#face_draw.cpp
#face_draw.h
#face_io.cpp
#face_io.h
#face_calc.cpp				
#face_calc.h
#face_results.cpp
#face_results.h
#cropped_frames.cpp			
#cropped_frames.h			
#face_tracker_adjustable_frame.cpp
				
#H_FILES = Makefile face_draw.h face_io.h face_results.h cropped_frames.h face_calc.h	
H_FILES =  face_draw.h face_io.h face_results.h cropped_frames.h face_calc.h	

all: peter_framing_filter 

clean:
	rm -f makehist *.o core

peter_framing_filter:	Makefile  face_draw.o face_io.o face_results.o face_calc.o cropped_frames.o face_tracker_adjustable_frame.o
	g++ ${CFLAGS} face_draw.o face_io.o face_results.o face_calc.o cropped_frames.o face_tracker_adjustable_frame.o ${LDFLAGS} -o peter_framing_filter${EXEEXT}

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

