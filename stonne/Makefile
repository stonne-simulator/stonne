CXX=g++
CXXFLAGS= -O3 -Iinclude/ -Iexternal/ -g #-ltcmalloc  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free
#DEBUGFLAGS=-D DEBUG_MEM_OUTPUT -D DEBUG_MSWITCH_FUNC
#-D DEBUG_MEM_OUTPUT -D DEBUG_MEM_INPUT -D DEBUG_ASWITCH_CONFIG -D DEBUG_ASWITCH_FUNC -D DEBUG_MSWITCH_CONFIG -D DEBUG_MSWITCH_FUNC
BIN = stonne

SOURCE =   $(wildcard src/*.cpp) \
           $(wildcard src/mRNA/*.cpp)
INCLUDES = $(wildcard include/*.h) \
		   $(wildcard include/mRNA/*.h)

OBJSDIR = objs
OBJS = $(patsubst src/%, $(OBJSDIR)/%, $(patsubst %.cpp,%.o,$(SOURCE)))





all: $(BIN)

$(BIN): $(OBJSDIR) $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS)  -o $@ $(OBJS)  #-pthread -ltcmalloc

$(OBJSDIR):
	mkdir -p $@ && mkdir -p $@/mRNA

$(OBJSDIR)/%.o: src/%.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c $< -o $@  #-ltcmalloc

.PHONY: clean
clean:
	rm -rf $(OBJSDIR) && rm stonne


