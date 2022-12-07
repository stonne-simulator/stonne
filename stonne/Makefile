CXX=g++
CXXFLAGS= -std=c++17 -O3 -Iinclude/ -Iexternal/ #-ltcmalloc  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
#DEBUGFLAGS= -O0 -g
#DEBUGFLAGS=-D DEBUG_MEM_OUTPUT -D DEBUG_MSWITCH_FUNC
#-D DEBUG_MEM_OUTPUT -D DEBUG_MEM_INPUT -D DEBUG_ASWITCH_CONFIG -D DEBUG_ASWITCH_FUNC -D DEBUG_MSWITCH_CONFIG -D DEBUG_MSWITCH_FUNC
BIN = stonne

SOURCE =   $(wildcard src/*.cpp) \
           $(wildcard src/TileGenerator/*.cpp) \
           $(wildcard src/TileGenerator/mRNA/*.cpp) \
           $(wildcard src/TileGenerator/StonneMapper/*.cpp) \
           $(wildcard src/TileGenerator/Utils/*.cpp)

INCLUDES = $(wildcard include/*.h) \
		   $(wildcard include/TileGenerator/*.h) \
		   $(wildcard include/TileGenerator/mRNA/*.h) \
		   $(wildcard include/TileGenerator/StonneMapper/*.h) \
		   $(wildcard include/TileGenerator/Utils/*.h)

OBJSDIR = objs
OBJS = $(patsubst src/%, $(OBJSDIR)/%, $(patsubst %.cpp,%.o,$(SOURCE)))





all: $(BIN)

$(BIN): $(OBJSDIR) $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS)  -o $@ $(OBJS)  #-pthread -ltcmalloc

$(OBJSDIR):
	mkdir -p $@ && \
	mkdir -p $@/TileGenerator && \
	mkdir -p $@/TileGenerator/mRNA && \
	mkdir -p $@/TileGenerator/StonneMapper && \
	mkdir -p $@/TileGenerator/Utils

$(OBJSDIR)/%.o: src/%.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c $< -o $@  #-ltcmalloc

.PHONY: clean
clean:
	rm -rf $(OBJSDIR) && rm stonne


