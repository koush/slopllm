export interface HeapAllocation {
  ptr: number;
  length: number;
}

export type HeapKey = number | object | symbol | undefined;

export class Heap {
  private static readonly ALIGNMENT = 256;
  private regions: HeapAllocation[] = [];
  private layoutBase: number | undefined;
  private layoutEnd: number | undefined;
  private allocationCount = 0;
  private allocationHash = 1469598103934665603n;

  manage(ptr: number, length: number): void {
    Heap.validateRegion(ptr, length);

    if (this.layoutBase === undefined) {
      this.layoutBase = ptr;
      this.layoutEnd = ptr;
    }

    const end = ptr + length;
    let index = 0;
    while (index < this.regions.length && this.regions[index].ptr < ptr) {
      index++;
    }

    const previous = this.regions[index - 1];
    const next = this.regions[index];
    if (previous && previous.ptr + previous.length > ptr) {
      throw new Error(`Heap region [${ptr}, ${end}) overlaps [${previous.ptr}, ${previous.ptr + previous.length})`);
    }
    if (next && end > next.ptr) {
      throw new Error(`Heap region [${ptr}, ${end}) overlaps [${next.ptr}, ${next.ptr + next.length})`);
    }

    const joinsPrevious = previous !== undefined && previous.ptr + previous.length === ptr;
    const joinsNext = next !== undefined && end === next.ptr;
    if (joinsPrevious && joinsNext) {
      previous.length += length + next.length;
      this.regions.splice(index, 1);
    } else if (joinsPrevious) {
      previous.length += length;
    } else if (joinsNext) {
      next.ptr = ptr;
      next.length += length;
    } else {
      this.regions.splice(index, 0, { ptr, length });
    }
  }

  alloc(size: number): HeapAllocation {
    const allocation = this.tryAlloc(size);
    if (allocation) return allocation;
    throw new Error(`Heap OOM: unable to allocate ${size} bytes`);
  }

  tryAlloc(size: number): HeapAllocation | undefined {
    if (!Number.isSafeInteger(size) || size <= 0) {
      throw new Error(`Heap allocation size must be a positive safe integer, got ${size}`);
    }

    for (let index = 0; index < this.regions.length; index++) {
      const region = this.regions[index];
      const alignmentOffset = (Heap.ALIGNMENT - region.ptr % Heap.ALIGNMENT) % Heap.ALIGNMENT;
      const ptr = region.ptr + alignmentOffset;
      const regionEnd = region.ptr + region.length;
      const requestedEnd = ptr + size;
      if (!Number.isSafeInteger(requestedEnd) || requestedEnd > regionEnd) continue;

      const alignedLength = Math.ceil(size / Heap.ALIGNMENT) * Heap.ALIGNMENT;
      const length = Math.min(alignedLength, regionEnd - ptr);
      const end = ptr + length;

      const prefixLength = ptr - region.ptr;
      const suffixLength = regionEnd - end;
      if (prefixLength && suffixLength) {
        region.length = prefixLength;
        this.regions.splice(index + 1, 0, { ptr: end, length: suffixLength });
      } else if (prefixLength) {
        region.length = prefixLength;
      } else if (suffixLength) {
        region.ptr = end;
        region.length = suffixLength;
      } else {
        this.regions.splice(index, 1);
      }

      this.layoutEnd = Math.max(this.layoutEnd ?? requestedEnd, requestedEnd);
      this.allocationCount++;
      this.allocationHash ^= BigInt(size);
      this.allocationHash = BigInt.asUintN(64, this.allocationHash * 1099511628211n);
      return { ptr, length };
    }

    return undefined;
  }

  claim(ptr: number, length: number): HeapAllocation | undefined {
    Heap.validateRegion(ptr, length);
    const end = ptr + length;
    for (let index = 0; index < this.regions.length; index++) {
      const region = this.regions[index];
      const regionEnd = region.ptr + region.length;
      if (ptr < region.ptr) return undefined;
      if (ptr >= regionEnd) continue;
      if (end > regionEnd) return undefined;

      const prefixLength = ptr - region.ptr;
      const suffixLength = regionEnd - end;
      if (prefixLength && suffixLength) {
        region.length = prefixLength;
        this.regions.splice(index + 1, 0, { ptr: end, length: suffixLength });
      } else if (prefixLength) {
        region.length = prefixLength;
      } else if (suffixLength) {
        region.ptr = end;
        region.length = suffixLength;
      } else {
        this.regions.splice(index, 1);
      }
      return { ptr, length };
    }
    return undefined;
  }

  contains(ptr: number, length: number): boolean {
    Heap.validateRegion(ptr, length);
    const end = ptr + length;
    return this.regions.some(region => ptr >= region.ptr && end <= region.ptr + region.length);
  }

  layoutSignature(): string {
    const offset = this.layoutBase === undefined ? 0 : (this.layoutEnd ?? this.layoutBase) - this.layoutBase;
    return `${offset}:${this.allocationCount}:${this.allocationHash.toString(16)}`;
  }

  describeRange(ptr: number, length: number): string {
    Heap.validateRegion(ptr, length);
    const end = ptr + length;
    const overlapping = this.regions.filter(region => ptr < region.ptr + region.length && end > region.ptr);
    const ranges = overlapping.length
      ? overlapping.map(region => `[0x${region.ptr.toString(16)},0x${(region.ptr + region.length).toString(16)})`).join(",")
      : "none";
    const freeBytes = this.regions.reduce((sum, region) => sum + region.length, 0);
    return `overlappingFree=${ranges} freeRegions=${this.regions.length} freeBytes=${freeBytes} layout=${this.layoutSignature()}`;
  }

  drainTo(destination: Heap): void {
    if (destination === this) return;

    // Build the destination state before mutating either heap so overlap errors
    // cannot leave a partially transferred source.
    const combined = [...destination.regions, ...this.regions]
      .map(region => ({ ...region }))
      .sort((a, b) => a.ptr - b.ptr);
    const merged: HeapAllocation[] = [];
    for (const region of combined) {
      const previous = merged[merged.length - 1];
      if (!previous) {
        merged.push(region);
        continue;
      }

      const previousEnd = previous.ptr + previous.length;
      if (previousEnd > region.ptr) {
        throw new Error(`Heap region [${region.ptr}, ${region.ptr + region.length}) overlaps [${previous.ptr}, ${previousEnd})`);
      }
      if (previousEnd === region.ptr) {
        previous.length += region.length;
      } else {
        merged.push(region);
      }
    }

    destination.regions = merged;
    if (destination.layoutBase === undefined && merged.length) {
      destination.layoutBase = merged[0].ptr;
      destination.layoutEnd = merged[0].ptr;
    }
    this.regions = [];
  }

  private static validateRegion(ptr: number, length: number): void {
    if (!Number.isSafeInteger(ptr) || ptr < 0) {
      throw new Error(`Heap pointer must be a non-negative safe integer, got ${ptr}`);
    }
    if (!Number.isSafeInteger(length) || length <= 0) {
      throw new Error(`Heap region length must be a positive safe integer, got ${length}`);
    }
    if (!Number.isSafeInteger(ptr + length)) {
      throw new Error(`Heap region end must be a safe integer, got ${ptr + length}`);
    }
  }
}
