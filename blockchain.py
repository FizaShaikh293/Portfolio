import hashlib
import json
import time
from datetime import datetime

class Block:
    """A block in the blockchain"""
    
    def __init__(self, index, miner_id, timestamp, transactions, proof, previous_hash):
        """Initialize a new Block"""
        self.index = index
        self.miner_id = miner_id
        self.timestamp = timestamp
        self.transactions = transactions
        self.proof = proof
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        """Calculate the hash of this block"""
        block_string = json.dumps({
            'index': self.index,
            'miner_id': self.miner_id,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'proof': self.proof,
            'previous_hash': self.previous_hash
        }, sort_keys=True).encode()
        
        return hashlib.sha256(block_string).hexdigest()
    
    def to_dict(self):
        """Convert the block to a dictionary"""
        return {
            'index': self.index,
            'miner_id': self.miner_id,
            'timestamp': self.timestamp,
            'transactions': self.transactions,
            'proof': self.proof,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }


class Blockchain:
    """A simple blockchain implementation for simulation purposes"""
    
    def __init__(self):
        """Initialize a new Blockchain with a genesis block"""
        self.chain = []
        self.current_transactions = []
        self.difficulty = 4
        
        # Create the genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the genesis block with a fixed previous hash"""
        genesis_block = Block(
            index=0,
            miner_id=0,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transactions=[],
            proof=0,
            previous_hash="0" * 64  # A string of 64 zeros
        )
        
        self.chain.append(genesis_block)
        return genesis_block
    
    def add_block(self, miner_id, proof):
        """Add a new block to the blockchain"""
        # Get previous block
        previous_block = self.get_latest_block()
        previous_hash = previous_block.hash
        
        # Create a new block
        block = Block(
            index=len(self.chain),
            miner_id=miner_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            transactions=self.current_transactions.copy(),
            proof=proof,
            previous_hash=previous_hash
        )
        
        # Reset the current list of transactions
        self.current_transactions = []
        
        # Add the block to the chain
        self.chain.append(block)
        
        return block
    
    def add_transaction(self, sender, recipient, amount):
        """Add a new transaction to the list of transactions"""
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return self.get_latest_block().index + 1
    
    def get_latest_block(self):
        """Return the latest block in the chain"""
        if not self.chain:
            return None
        return self.chain[-1]
    
    def is_chain_valid(self):
        """Check if the blockchain is valid"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check if the hash of the block is correct
            if current.hash != current.calculate_hash():
                return False
            
            # Check if the previous hash reference is correct
            if current.previous_hash != previous.hash:
                return False
        
        return True