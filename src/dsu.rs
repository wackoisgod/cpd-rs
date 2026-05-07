//! Disjoint Set Union (Union-Find) with path compression and union-by-rank.
//! Used for primitive identity: each face's representative root is the id of
//! the primitive currently subsuming it.

pub struct Dsu {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Dsu {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: u32) -> u32 {
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // path compression
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    /// Union the sets containing a and b. Returns the new root.
    pub fn union(&mut self, a: u32, b: u32) -> u32 {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return ra;
        }
        let (winner, loser) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            (ra, rb)
        } else {
            self.rank[ra as usize] += 1;
            (ra, rb)
        };
        self.parent[loser as usize] = winner;
        winner
    }

    /// Pre-find variant useful when caller already has both roots.
    pub fn union_roots(&mut self, ra: u32, rb: u32) -> u32 {
        if ra == rb {
            return ra;
        }
        let (winner, loser) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else if self.rank[ra as usize] > self.rank[rb as usize] {
            (ra, rb)
        } else {
            self.rank[ra as usize] += 1;
            (ra, rb)
        };
        self.parent[loser as usize] = winner;
        winner
    }

    pub fn is_root(&self, x: u32) -> bool {
        self.parent[x as usize] == x
    }

    /// Force a directional link: `loser_root` becomes a child of
    /// `winner_root`. Caller must already have both as roots.
    pub fn link(&mut self, winner_root: u32, loser_root: u32) {
        debug_assert!(self.is_root(winner_root));
        debug_assert!(self.is_root(loser_root));
        if winner_root != loser_root {
            self.parent[loser_root as usize] = winner_root;
            // bump rank if equal so future paths stay shallow
            if self.rank[winner_root as usize] == self.rank[loser_root as usize] {
                self.rank[winner_root as usize] = self.rank[winner_root as usize].saturating_add(1);
            }
        }
    }
}
