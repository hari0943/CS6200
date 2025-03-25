#include <iostream>
#include <vector>
#include "simulator.h"

Bus::Bus(std::vector<Cache*>& _caches) {
	caches = _caches;
	shared_line = false;
	num_busrd = 0;
	num_busrdx = 0;
	num_flushes = 0;
}

bool Bus::getSharedLine(){
	return shared_line;
}

void Bus::setSharedLine(){
	shared_line = true;
}

void Bus::sendMessage(BusRequest request, unsigned long long block_address, int sender_cache_id) {
	// This function invokes handleBusRequest with the given request on
	// all core's caches except the one that sent the request

	switch (request) {
		case BusRequest::BusRd:
			num_busrd++;
			break;
		case BusRequest::BusRdX:
			num_busrdx++;
			break;
		case BusRequest::Flush:
			num_flushes++;
			break;
	}
	if (request == BusRequest::Flush) {
		// We just simulate writing back to memory
		// Since cache to cache sharing is disabled, there is no need to invoke 
		// handleBusRequest for a Flush request
		return;
	}

	// Unset shared_line before sending Bus request to all other cores
	shared_line = false;

	for (int i=0; i<caches.size(); i++) {
		if (caches[i]->getId() != sender_cache_id) {
			caches[i]->handleBusRequest(request, block_address);
		}
	}
}

void Bus::printStats() {
	std::cout << ">> Bus stats" << std::endl;
	std::cout << "Number of BusRd   : " << num_busrd << std::endl;
	std::cout << "Number of BusRdX  : " << num_busrdx << std::endl;
	std::cout << "Number of Flushes : " << num_flushes << std::endl;
}
