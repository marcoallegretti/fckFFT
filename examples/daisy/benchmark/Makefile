TARGET = benchmark

CPP_SOURCES = benchmark.cpp

LIBDAISY_DIR ?= ../../../../libDaisy
FCKFFT_DIR ?= ../../..

C_INCLUDES += -I$(FCKFFT_DIR)/include

ifeq ($(BOARD),patch_sm)
C_DEFS += -DFCKFFT_PATCH_SM
endif

SYSTEM_FILES_DIR = $(LIBDAISY_DIR)/core
include $(SYSTEM_FILES_DIR)/Makefile
