#include <iostream>
#include <list>
#include "simulator.h"

CacheBlock::CacheBlock(int _tag, CacheBlockState _state) {
	tag = _tag;
	state = _state;
}

CacheSet::CacheSet() {
	for (int i=0; i < ASSOCIATIVITY; i++) {
		blocks.push_back(CacheBlock(0, CacheBlockState::Invalid));
	}
}

CacheBlockState CacheSet::getState(unsigned long long tag) {
	for (std::list<CacheBlock>::iterator iter=blocks.begin(); iter != blocks.end() ; iter++) {
		if (iter->state != CacheBlockState::Invalid && iter->tag == tag) {
			return iter->state;
		}
	}
	return CacheBlockState::Invalid;
}

void CacheSet::moveToMRU(unsigned long long tag) {
	for (std::list<CacheBlock>::iterator iter=blocks.begin(); iter != blocks.end() ; iter++) {
		if (iter->state != CacheBlockState::Invalid && iter->tag == tag) {
			// Move to MRU position
			CacheBlock cb = *iter;
			blocks.erase(iter);
			blocks.push_back(cb);
		}
	}
}

CacheBlock CacheSet::insertCacheBlock(CacheBlock new_block) {
	CacheBlock evicted_block = blocks.front();
	blocks.pop_front();
	blocks.push_back(new_block);
	return evicted_block;
}

void CacheSet::setState(unsigned long long tag, CacheBlockState state) {
	for (std::list<CacheBlock>::iterator iter=blocks.begin(); iter != blocks.end() ; iter++) {
		if (iter->state != CacheBlockState::Invalid && iter->tag == tag) {
			if (state == CacheBlockState::Invalid) {
				// Move to LRU position
				// We want invalid blocks to be at the LRU position so that
				// we don't evict a valid block when an invalid block is available
				CacheBlock cb = *iter;
				cb.state = state;
				cb.tag = 0;
				blocks.erase(iter);
				blocks.push_front(cb);
			} else {
				iter->state = state;
			}
		}
	}
}

void CacheSet::print() {
	for (std::list<CacheBlock>::iterator iter = blocks.begin(); iter != blocks.end(); iter++) {
		switch (iter->state) {
			case CacheBlockState::Modified:
				std::cout << "M";
				break;
			case CacheBlockState::Exclusive:
				std::cout << "E";
				break;
			case CacheBlockState::Shared:
				std::cout << "S";
				break;
			case CacheBlockState::Invalid:
				std::cout << "I";
				break;
		}
		std::cout << ":" << "0x" << std::hex << iter->tag;
		std::cout << "\t";
	}
	std::cout << std::endl;
}
