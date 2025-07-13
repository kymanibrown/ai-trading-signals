#!/usr/bin/env python3
"""
A simple Python file with basic examples.
"""

def greet(name="World"):
    """Return a greeting message."""
    return f"Hello, {name}!"

def calculate_sum(numbers):
    """Calculate the sum of a list of numbers."""
    return sum(numbers)

def main():
    """Main function to demonstrate basic Python functionality."""
    print("=== Python File Example ===")
    
    # Basic greeting
    print(greet())
    print(greet("Python"))
    
    # List operations
    numbers = [1, 2, 3, 4, 5]
    print(f"Numbers: {numbers}")
    print(f"Sum: {calculate_sum(numbers)}")
    
    # Dictionary example
    person = {
        "name": "Alice",
        "age": 30,
        "city": "New York"
    }
    print(f"Person: {person}")
    
    # List comprehension
    squares = [x**2 for x in range(1, 6)]
    print(f"Squares: {squares}")

if __name__ == "__main__":
    main() 