export interface Allocator {
    alloc(size: number): number;
    free(ptr: number): void;
}

export class ArenaAllocator implements Allocator {
    private offset = 0;
    private allocationCount = 0;
    private allocationHash = 1469598103934665603n;
    private static readonly ALIGN = 256;

    constructor(public readonly base: number, public readonly size: number) {
    }

    alloc(size: number): number {
        const aligned = Math.ceil(this.offset / ArenaAllocator.ALIGN) * ArenaAllocator.ALIGN;
        if (aligned + size > this.size) throw new Error(`ArenaAllocator OOM: need ${aligned + size}, have ${this.size}`);
        this.offset = aligned + size;
        this.allocationCount++;
        this.allocationHash ^= BigInt(size);
        this.allocationHash = BigInt.asUintN(64, this.allocationHash * 1099511628211n);
        return this.base + aligned;
    }

    layoutSignature(): string {
        return `${this.offset}:${this.allocationCount}:${this.allocationHash.toString(16)}`;
    }

    free(_ptr: number): void {}
}
