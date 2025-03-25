#pragma once
#include <vector>
#include <list>

/*****************************************************************************
 *                           Simulator parameters                            *
 *****************************************************************************/
#define NUMBER_OF_CORES 4
#define SET_BITS 2
#define NUMBER_OF_SETS (1<<SET_BITS)
#define ASSOCIATIVITY_BITS 2
#define ASSOCIATIVITY (1<<ASSOCIATIVITY_BITS)
#define CACHE_OFFSET_BITS 6
#define CACHE_BLOCK_SIZE (1<<CACHE_OFFSET_BITS)

/*****************************************************************************
 *                           Enums used throughout                           *
 *****************************************************************************/

typedef enum {
	BusRd,
	BusRdX,
	Flush,
} BusRequest;

typedef enum {
	ProcRd,
	ProcWr
} ProcRequest;

typedef enum {
	MSI,
	MESI
} Protocol;

typedef enum {
	Modified,
	Exclusive,
	Shared,
	Invalid
} CacheBlockState;

/*****************************************************************************
 *                  Classes representing Cache and Bus                       *
 *****************************************************************************/
class CacheBlock {
	public:
		unsigned long long tag;
		CacheBlockState state;

		CacheBlock(int _tag, CacheBlockState _state);
};

class CacheSet {
	public:
		std::list<CacheBlock> blocks;

		CacheSet();

		// Returns the state of the Block with tag given
		// Returns CacheBlockState::Invalid if cache block is not found
		CacheBlockState getState(unsigned long long tag);

		// Sets the state of the block with tag given to state
		void setState(unsigned long long tag, CacheBlockState state);

		// Moves the block with tag given to MRU position of the set
		void moveToMRU(unsigned long long tag);

		// Inserts a new cache block in the set
		CacheBlock insertCacheBlock(CacheBlock new_block);

		// Prints the cache set
		void print();
};

class Bus;

class Cache {
	public:
		// Cache ID, used when sending request on the Bus
		int id;

// Protocol used: MSI or MESI
		Protocol protocol;

		// CacheSets
		std::vector<CacheSet> sets;

		// Pointer to the shared Bus
		Bus* bus;

		// Counters
		int num_reads, num_read_misses, num_writes, num_write_misses, num_writebacks, num_invalidations;

		Cache(int _id, Protocol protocol);
		void setBus(Bus* _bus);

		// Inserts a new block with the block_address in the state that is provided
		// The new block will be in the MRU position in its set
		CacheBlock insertCacheBlock(unsigned long long block_address, CacheBlockState state);

		// Returns the state of a cache block
		// Returns CacheBlockState::Invalid if the block is not present in the cache
		CacheBlockState getState(unsigned long long block_address);

		// Sets the state of a block with the address block_address to state provided
		// This WILL NOT move the block to MRU position
		void setState(unsigned long long block_address, CacheBlockState state);

		// Moves the cache block with address block_address to the MRU position in its set
		void moveToMRU(unsigned long long block_address);

		// Returns the cache ID
		int getId();

		// Returns the protocol being used
		Protocol getProtocol();

		// This function will handle the Bus requests to the block_address
		void handleBusRequest(BusRequest request, unsigned long long block_address);

		// This function will handle the Processor requests to the address
		// Note that this address include the cache offset
		void handleProcRequest(ProcRequest request, unsigned long long address);

		// Prints the cache statistics
		void printStats();
};

class Bus {
	public:
		std::vector<Cache*> caches;
		bool shared_line;

		// Counters for different request types
		int num_busrd, num_busrdx, num_flushes;

		Bus(std::vector<Cache*>& _caches);

		// Returns the value of the shared line
		bool getSharedLine();

		// Sets the value of the shared line to true
		void setSharedLine();

		// Invokes handleBusRequest on all other caches except the one that sent the request
		// Flush requests are only counted and will not be forwarded to other caches
		void sendMessage(BusRequest request, unsigned long long block_address, int sender_cache_id);

		// Prints the counts
		void printStats();
};
