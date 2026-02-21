# ariannamethod — field physics engine
# builds libaml.so (Linux) or libaml.dylib (macOS)
# with optional BLAS acceleration for notorch

UNAME := $(shell uname)

CC = gcc
CFLAGS = -O2 -fPIC -Wall

ifeq ($(UNAME), Darwin)
	# macOS: Apple Accelerate (zero deps, AMX/Neural Engine)
	TARGET = libaml.dylib
	LDFLAGS = -shared -framework Accelerate
	CFLAGS += -DUSE_BLAS -DACCELERATE
else
	# Linux: OpenBLAS (apt install libopenblas-dev)
	TARGET = libaml.so
	BLAS_CHECK := $(shell pkg-config --exists openblas 2>/dev/null && echo yes)
	ifeq ($(BLAS_CHECK), yes)
		LDFLAGS = -shared $(shell pkg-config --libs openblas)
		CFLAGS += -DUSE_BLAS $(shell pkg-config --cflags openblas)
	else
		# fallback: try -lopenblas directly
		LDFLAGS_BLAS := $(shell echo 'int main(){}' | gcc -xc - -lopenblas -o /dev/null 2>/dev/null && echo "-lopenblas")
		ifneq ($(LDFLAGS_BLAS),)
			LDFLAGS = -shared -lopenblas
			CFLAGS += -DUSE_BLAS
		else
			LDFLAGS = -shared -lm
		endif
	endif
endif

SRCS = ariannamethod.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c ariannamethod.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

test: $(TARGET)
	$(CC) $(CFLAGS) -o test_aml test_aml.c -L. -laml -lm
	LD_LIBRARY_PATH=. ./test_aml

.PHONY: all clean test
