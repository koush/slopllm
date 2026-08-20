import { Tensor } from "./tensor";

export type TensorTree = Tensor | TensorTree[] | string | number | boolean | null | undefined | void | { [name: string]: TensorTree };

export function mapTensors(tree: TensorTree, mapTensor: (tensor: Tensor) => Tensor): TensorTree {
  if (tree instanceof Tensor) {
    return mapTensor(tree);
  }
  if (Array.isArray(tree)) {
    return tree.map(item => mapTensors(item, mapTensor));
  }
  if (tree && typeof tree === "object") {
    const result: Record<string, TensorTree> = {};
    for (const [key, item] of Object.entries(tree)) {
      result[key] = mapTensors(item, mapTensor);
    }
    return result;
  }
  return tree;
}

export function collectTensors(tree: TensorTree): Set<Tensor> {
  const tensors = new Set<Tensor>();
  mapTensors(tree, tensor => {
    tensors.add(tensor);
    return tensor;
  });
  return tensors;
}
