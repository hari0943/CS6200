#include <iostream>
#include <vector>
#include <list>
#include "simulator.h"
using namespace std;

int main() {
	Protocol protocol;
	string protocolName;
	cin >> protocolName;
	if (protocolName == "MESI") {
		protocol = Protocol::MESI;
		cout << "Protocol Used : MESI" << endl;
	} else if (protocolName == "MSI") {
		protocol = Protocol::MSI;
		cout << "Protocol Used : MSI" << endl;
	} else {
		exit(0);
	}

	vector<Cache*> caches;
	for (int i=0; i < NUMBER_OF_CORES; i++) {
		caches.push_back(new Cache(i, protocol));
	}

	Bus bus(caches);
	for (int i=0; i < NUMBER_OF_CORES; i++) {
		caches[i]->setBus(&bus);
	}

	int core;
	char r_or_w;
	unsigned long long address;
	while (true) {
		cin >> core;
		if (core == -1) {
			break;
		}
		cin >> r_or_w >> hex >> address;
		if (core >= NUMBER_OF_CORES || core < 0) {
			cout << "Incorrect core number " << core << endl;
			exit(0);
		}
		if (r_or_w == 'r') {
			caches[core]->handleProcRequest(ProcRequest::ProcRd, address);
		} else if (r_or_w == 'w') {
			caches[core]->handleProcRequest(ProcRequest::ProcWr, address);
		}
	}

	// Print the statistics and contents of cache
	bus.printStats();
	for (int i=0; i < NUMBER_OF_CORES; i++) {
		caches[i]->printStats();
	}
}
