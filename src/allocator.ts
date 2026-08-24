export interface Allocator {
    alloc(size: number): number;
    free(ptr: number): void;
}

export class ArenaAllocator implements Allocator {
    private offset = 0;
    private static readonly ALIGN = 256;

    constructor(public readonly base: number, public readonly size: number) {
    }

    alloc(size: number): number {
        const aligned = Math.ceil(this.offset / ArenaAllocator.ALIGN) * ArenaAllocator.ALIGN;
        if (aligned + size > this.size) throw new Error(`ArenaAllocator OOM: need ${aligned + size}, have ${this.size}`);
        this.offset = aligned + size;
        return this.base + aligned;
    }

    free(_ptr: number): void {}
}
