use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
struct VectorBlock {
    raw: Vec<f32>,
    n: usize,
    dim: usize
}

impl VectorBlock {
    fn new(n: usize, dim: usize) -> VectorBlock {
	return VectorBlock {
	    raw: vec![0.0; n * dim],
	    n: n,
	    dim: dim
	};
    }

    fn set(&mut self, i: usize, v: &Vec<f32>) {
	let start = i * self.dim;
	let end = (i + 1) * self.dim;
	self.raw[start..end].copy_from_slice(v);
    }

    fn get(&self, i: usize) -> &[f32] {
	let start = i * self.dim;
	let end = (i + 1) * self.dim;
	return &self.raw[start .. end];
    }

    fn euclidean(&self, i: usize, v: &Vec<f32>) -> f32 {
	let start = i * self.dim;
	let mut dist = 0.0;
	for j in 0 .. self.dim {
	    dist += (v[j] - self.raw[start + j]).powf(2.0);
	}
	return dist.sqrt();
    }
}

#[derive(Debug)]
struct Vectors {
    vectors: Vec<VectorBlock>,   
    block_size: usize,
    dim: usize,
    pos: usize
}

impl Vectors {

    fn new(blksze: usize, dim: usize) -> Vectors {
	return Vectors {
	    vectors: vec![VectorBlock::new(blksze, dim)],
	    block_size: blksze,
	    dim: dim,
	    pos: 0
	};
    }

    fn capacity(&self) -> usize {
	return self.vectors.len() * self.block_size;
    }

    fn n_filled(&self) -> usize {
	return (self.vectors.len() - 1) * self.block_size;
    }
    
    fn insert(&mut self, v: &Vec<f32>) {
	if self.pos >=  self.capacity() {
	    self.vectors.push(VectorBlock::new(self.block_size, self.dim));
	}
	let pos_of_vec = self.vectors.len() - 1;
	let pos_in_block = self.pos - self.n_filled();	
	self.vectors[pos_of_vec].set(pos_in_block, v);
	self.pos += 1;
    }
    
    fn get(&self, id: usize) -> &[f32] {
	let block = id / self.block_size;
	let pos = id - (block * self.block_size);
	return self.vectors[block].get(pos);
    }

    fn euclidean(&self, id: usize, v: &Vec<f32>) -> f32 {
	let block = id / self.block_size;
	let pos = id - (block * self.block_size);
	return self.vectors[block].euclidean(pos, v);	
    }

}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Neighbor {
    dst: usize,
    weight: f32
}

impl Neighbor {
    fn new(dst: usize, weight: f32) -> Neighbor {
	return Neighbor {
	    dst: dst,
	    weight: weight
	};
    }
}


impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return other.weight.partial_cmp(&self.weight);
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
	return other.partial_cmp(self).unwrap();
    }
}

impl Eq for Neighbor {}


#[derive(Debug)]
struct HNSWG {
    node_vec: Vectors,
    layer_edges: Vec<HashMap<usize, Vec<usize>>>
}

impl HNSWG {
    fn new(block_size: usize, dim: usize, layers: usize) -> HNSWG {
	return HNSWG {
	    node_vec: Vectors::new(block_size, dim),
	    layer_edges: vec![HashMap::new(); layers]
	};
    }

    fn add_edge(&mut self, layer: usize, from: usize, to: usize) {
	if !self.layer_edges[layer].contains_key(&from) {
	    self.layer_edges[layer].insert(from, Vec::new());
	}
	self.layer_edges[layer].get_mut(&from).unwrap().push(to);
    }
    
    fn add_vector(&mut self, v: &Vec<f32>) {
	self.node_vec.insert(v);
    }

    fn get_vector(&self, node_id: usize) -> &[f32] {
	return self.node_vec.get(node_id);
    }

    fn search_layer(&self,
		    query: &Vec<f32>,
		    entry_points: BinaryHeap<Neighbor>,
		    k_neighbors: usize,
		    layer: usize
    ) -> BinaryHeap<Neighbor> {
	let mut candidates = BinaryHeap::new();
	let mut visited = HashSet::new();
	let mut nearest_neighbors = entry_points.clone();
	for &candidate in nearest_neighbors.iter() {
	    visited.insert(candidate.dst);
	    candidates.push(Neighbor::new(candidate.dst, -candidate.weight));
	}
	
	while candidates.len() > 0 {
	    let mref = nearest_neighbors.peek().unwrap();
	    let candidate = candidates.pop().unwrap();
	    if candidate.weight > -mref.weight {
		break;
	    }
	    if self.layer_edges[layer].contains_key(&candidate.dst) {
		let neighbors = &self.layer_edges[layer][&candidate.dst];
		for &e in neighbors {
		    if !visited.contains(&e) {
			visited.insert(e);
			let dist = self.node_vec.euclidean(e, query);
			let bsf = -nearest_neighbors.peek().unwrap().weight;
			if dist < bsf || nearest_neighbors.len() < k_neighbors {
			    candidates.push(Neighbor::new(e, dist));
			    nearest_neighbors.push(Neighbor::new(e, -dist));
			    if nearest_neighbors.len() > k_neighbors {
				nearest_neighbors.pop();
			    }
			}
		    }
		}
	    }
	}
	return nearest_neighbors;
    }
}


#[cfg(test)]
mod tests {
    use super::*; 

    #[test]
    fn test_search_layer() {
	let mut candidates = BinaryHeap::new();
	let mut hnswg = HNSWG::new(3, 3, 2);
	
	for i in 0 .. 10 {
	    hnswg.add_vector(&vec![1.0 * i as f32; 3]);	
	}

	hnswg.add_edge(0, 0, 1);
	hnswg.add_edge(0, 0, 5);
	hnswg.add_edge(0, 1, 4);
	hnswg.add_edge(0, 1, 3);
	hnswg.add_edge(0, 1, 7);
	
	let query = vec![1.5 as f32; 3];
	let dist = hnswg.node_vec.euclidean(0, &query);

	candidates.push(Neighbor::new(0, -dist));

	let mut result = hnswg.search_layer(
	    &query, candidates, 3, 0);
	
	let x = vec![0, 3, 1];
	let mut i = 0;
	while result.len() > 0 { 
	    let r = result.pop();
	    i += 1;
         }
    }
    
}


fn main() {
    let mut x = Vectors::new(3, 4);
    for i in 0 .. 10 {
	x.insert(&vec![1.0 * i as f32; 4]);	
    }

    let y = vec![10.0; 4];
    println!("Hello, world!");
    println!("{:?}", x);
    let mut h = BinaryHeap::new();
    for i in 0 .. 10 {
	let d = x.euclidean(i, &y);
	println!("{:?} - {:?} = {}", x.get(i), y, x.euclidean(i, &y));	
	h.push(Neighbor::new(0, -d));
    }
    while h.len() > 0 {
	let e = h.pop().unwrap();
	println!("{:?}", e);
    }
}

