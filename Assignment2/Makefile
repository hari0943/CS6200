OBJS = bus.o cache.o cacheset.o main.o

sim: coherence.o $(OBJS)
	g++ $(OBJS) coherence.o -o sim

$(OBJS): bus.cpp cache.cpp cacheset.cpp main.cpp
	g++ -c bus.cpp cache.cpp cacheset.cpp main.cpp

coherence.o: coherence.cpp
	g++ -c coherence.cpp

clean:
	rm *.o sim
