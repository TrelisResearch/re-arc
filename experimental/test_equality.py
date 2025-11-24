# Quick test to verify tuple equality checks all cells
grid1 = ((1, 2, 3), (4, 5, 6))
grid2 = ((1, 2, 3), (4, 5, 6))
grid3 = ((1, 2, 3), (4, 5, 7))  # Different last cell

print(f"grid1 == grid2: {grid1 == grid2}")  # Should be True
print(f"grid1 == grid3: {grid1 == grid3}")  # Should be False
print(f"grid1 is grid2: {grid1 is grid2}")  # Should be False (different objects)
