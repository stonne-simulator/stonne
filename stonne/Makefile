CXX=g++
CXXFLAGS= -O3 -Iinclude/ -Iexternal/ #-ltcmalloc  -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free 
#DEBUGFLAGS=-D DEBUG_MEM_OUTPUT -D DEBUG_MSWITCH_FUNC
#-D DEBUG_MEM_OUTPUT -D DEBUG_MEM_INPUT -D DEBUG_ASWITCH_CONFIG -D DEBUG_ASWITCH_FUNC -D DEBUG_MSWITCH_CONFIG -D DEBUG_MSWITCH_FUNC
BIN=stonne
SOURCE=$(wildcard src/*.cpp)
OBJSDIR=objs
OBJS=$(patsubst src/%,$(OBJSDIR)/%,$(patsubst %.cpp,%.o,$(SOURCE)))
INCLUDES=$(wildcard include/*.h)




all: $(BIN)

$(BIN): $(OBJSDIR) $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS)  -o $@ $(OBJS)  #-pthread -ltcmalloc

$(OBJSDIR):
	mkdir -p $@

$(OBJSDIR)/%.o: src/%.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c $< -o $@  #-ltcmalloc



clean:
	rm -rf $(OBJSDIR) && rm stonne


