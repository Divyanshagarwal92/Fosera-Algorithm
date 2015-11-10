CFLAGS = `pkg-config --cflags opencv`
C++11 = -std=c++11
LIBS = `pkg-config --libs opencv`

% : %.cpp
	g++ $(CFLAGS) $(LIBS) $(C++11) -o $@ $<
