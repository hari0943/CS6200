#include "simulator.h"
//#include <iostream>

void Cache::handleBusRequest(BusRequest request, unsigned long long block_address){
	CacheBlockState currentCacheBlockState=Cache::getState(block_address);
	switch(Cache::protocol){
		case Protocol::MESI:
			switch(currentCacheBlockState){
				case CacheBlockState::Invalid:
					//do nothing
					break;
				case CacheBlockState::Shared:
					switch(request){
						case BusRequest::BusRd:
							//Set shared bit
							Cache::bus->setSharedLine();
							break;
						case BusRequest::BusRdX:
							//Tansition to invalid
							Cache::setState(block_address,CacheBlockState::Invalid);
							Cache::num_invalidations++;
							break;
					}
					break;
				case CacheBlockState::Exclusive:
					switch(request){
						case BusRequest::BusRd:
							//Transition to shared
							//Set shared bit
							Cache::setState(block_address,CacheBlockState::Shared);
							Cache::bus->setSharedLine();
							break;
						case BusRequest::BusRdX:
							//Transition to invalid
							Cache::setState(block_address,CacheBlockState::Invalid);
							Cache::num_invalidations++;
							break;
					}
					break;
				case CacheBlockState::Modified:
					switch(request){
						case BusRequest::BusRd:
							//Transition to shared
							//Flush
							//Set Shared Bit
							Cache::setState(block_address,CacheBlockState::Shared);
							Cache::bus->sendMessage(BusRequest::Flush,block_address,Cache::id);
							Cache::bus->setSharedLine();
							Cache::num_writebacks++;
							break;
						case BusRequest::BusRdX:
							//Transition to invalid
							//Flush
							Cache::setState(block_address,CacheBlockState::Invalid);
							Cache::bus->sendMessage(BusRequest::Flush,block_address,Cache::id);
							Cache::num_invalidations++;
							Cache::num_writebacks++;
							break;
					}
					break;
			}
			break;
		case Protocol::MSI:
			switch(currentCacheBlockState){
				case CacheBlockState::Invalid:
					//do nothing
					break;
				case CacheBlockState::Shared:
					switch(request){
						case BusRequest::BusRd:
							//set shared bit
							Cache::bus->setSharedLine();
							break;
						case BusRequest::BusRdX:
							Cache::setState(block_address,CacheBlockState::Invalid);
							Cache::num_invalidations++;
							break;
					}
					break;
				case CacheBlockState::Modified:
					switch(request){
						case BusRequest::BusRd:
							//Transition to shared
							//Flush
							//Set shared bit
							Cache::setState(block_address,CacheBlockState::Shared);
							Cache::bus->sendMessage(BusRequest::Flush,block_address,Cache::id);
							Cache::bus->setSharedLine();
							Cache::num_writebacks++;
							break;
						case BusRequest::BusRdX:
							//Transition to invalid
							//Flush
							Cache::setState(block_address,CacheBlockState::Invalid);
							Cache::bus->sendMessage(BusRequest::Flush,block_address,Cache::id);
							Cache::num_invalidations++;
							Cache::num_writebacks++;
							break;
					}
					break;
			}

	}
}

unsigned long long getTagNumber(unsigned long long address){
	return address>>(SET_BITS+ASSOCIATIVITY_BITS+CACHE_OFFSET_BITS);
}
unsigned long long getBlockAddress(unsigned long long address){
	return address>>(CACHE_OFFSET_BITS);
}
unsigned long long getBlockOffset(unsigned long long address){
	return address&((1<<CACHE_OFFSET_BITS)-1);
}
void Cache::handleProcRequest(ProcRequest request, unsigned long long address) {
	unsigned long long int blockAddress=getBlockAddress(address);
	CacheBlock evictedBlock(0, CacheBlockState::Invalid);
	CacheBlockState currentCacheBlockState=Cache::getState(blockAddress);	
	switch(Cache::protocol){
		case Protocol::MSI:
			switch(currentCacheBlockState){
				case CacheBlockState::Invalid:
					switch(request){
						case ProcRequest::ProcRd:
							//Issue BusRd
							//Set Block to shared
							//Write back evicted modified block
							Cache::bus->sendMessage(BusRequest::BusRd,blockAddress,Cache::getId());
							evictedBlock=Cache::insertCacheBlock(blockAddress,CacheBlockState::Shared);
							if(evictedBlock.state==CacheBlockState::Modified){
								Cache::bus->sendMessage(BusRequest::Flush,evictedBlock.tag<<SET_BITS|(blockAddress&((1<<SET_BITS)-1)),Cache::getId());
								Cache::num_writebacks++;
							}
							Cache::num_reads++;
							Cache::num_read_misses++;
							break;
						case ProcRequest::ProcWr:
							//Issue BusRdx on bus
							//transition to modified
							//others on seeing BusRdx invalidate
							Cache::bus->sendMessage(BusRequest::BusRdX,blockAddress,Cache::getId());
							evictedBlock=Cache::insertCacheBlock(blockAddress,CacheBlockState::Modified);
							if(evictedBlock.state==CacheBlockState::Modified){
								Cache::bus->sendMessage(BusRequest::Flush,evictedBlock.tag<<SET_BITS|(blockAddress&((1<<SET_BITS)-1)),Cache::getId());
								Cache::num_writebacks++;
							}
							//
							Cache::num_writes++;
							Cache::num_write_misses++;
							break;
					}
					break;
				case CacheBlockState::Shared:
					switch(request){
						case ProcRequest::ProcRd:
							//change MRU
							Cache::moveToMRU(blockAddress);
							Cache::num_reads++;
							break;
						case ProcRequest::ProcWr:
							//Issue BusRdX
							//Change state to Modified
							//Change MRU
							Cache::bus->sendMessage(BusRequest::BusRdX,blockAddress,Cache::getId());
							Cache::setState(blockAddress,CacheBlockState::Modified);
							Cache::moveToMRU(blockAddress);
							Cache::num_writes++;
							break;
					}
					break;
				case CacheBlockState::Modified:
					switch(request){
						case ProcRequest::ProcRd:
							//moveMRU
							Cache::moveToMRU(blockAddress);
							Cache::num_reads++;
							break;
						case ProcRequest::ProcWr:
							//moveMRU
							Cache::moveToMRU(blockAddress);
							Cache::num_writes++;
							break;
					}
					break;
			}
			break;
		case Protocol::MESI:
			//std::cout<<"Handling a Proc request by Core"<<Cache::id<<" for "<<address<<"\n";
			//std::cout<<"Block Status "<<currentCacheBlockState<<"\n";
			switch(currentCacheBlockState){
				case CacheBlockState::Invalid:
					switch(request){
						case ProcRequest::ProcRd:
							//Issue BusRead
							//Set shared line
							//If others have copy, set to shared, else put as exclusive
							Cache::bus->sendMessage(BusRequest::BusRd,blockAddress,Cache::getId());
							if(Cache::bus->getSharedLine()){
								//Someone has the line
								evictedBlock=Cache::insertCacheBlock(blockAddress,CacheBlockState::Shared);
							}else{
								//Nobody has it
								evictedBlock=Cache::insertCacheBlock(blockAddress,CacheBlockState::Exclusive);
							}
							if(evictedBlock.state==CacheBlockState::Modified){
								Cache::bus->sendMessage(BusRequest::Flush,evictedBlock.tag<<SET_BITS|(blockAddress&((1<<SET_BITS)-1)),Cache::getId());
								Cache::num_writebacks++;
							}
							Cache::num_reads++;
							Cache::num_read_misses++;
							break;
						case ProcRequest::ProcWr:
							//Issue BusRdx on bus
							//transition to modified
							//others on seeing BusRdx invalidate
							Cache::bus->sendMessage(BusRequest::BusRdX,blockAddress,Cache::getId());
							evictedBlock=Cache::insertCacheBlock(blockAddress,CacheBlockState::Modified);
							if(evictedBlock.state==CacheBlockState::Modified){
								Cache::bus->sendMessage(BusRequest::Flush,evictedBlock.tag<<SET_BITS|(blockAddress&((1<<SET_BITS)-1)),Cache::getId());
								Cache::num_writebacks++;
							}
							Cache::num_writes++;
							Cache::num_write_misses++;
							break;
					}
					break;
				case CacheBlockState::Shared:
					switch(request){
						case ProcRequest::ProcRd:
							//change MRU
							Cache::moveToMRU(blockAddress);
							Cache::num_reads++;
							break;
						case ProcRequest::ProcWr:
							//Issue BusRdX
							//Change state to Modified
							//Change MRU
							Cache::bus->sendMessage(BusRequest::BusRdX,blockAddress,Cache::getId());
							Cache::setState(blockAddress,CacheBlockState::Modified);
							Cache::moveToMRU(blockAddress);
							Cache::num_writes++;
							break;
					}
					break;
				case CacheBlockState::Exclusive:
					switch(request){
						case ProcRequest::ProcRd:
							//Change MRU
							Cache::moveToMRU(blockAddress);
							Cache::num_reads++;
							break;
						case ProcRequest::ProcWr:
							//change to modified
							//Change MRU
							Cache::setState(blockAddress,CacheBlockState::Modified);
							Cache::moveToMRU(blockAddress);
							Cache::num_writes++;
							break;
					}		
					break;
				case CacheBlockState::Modified:
					switch(request){
						case ProcRequest::ProcRd:
							//moveMRU
							Cache::moveToMRU(blockAddress);
							Cache::num_reads++;
							break;
						case ProcRequest::ProcWr:
							//moveMRU
							Cache::moveToMRU(blockAddress);
							Cache::num_writes++;
							break;
					}
					break;
			}
			break;
	}
}
