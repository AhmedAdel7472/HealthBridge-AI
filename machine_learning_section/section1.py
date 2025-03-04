# 1 - Count vowels in a string
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

# 2 - Generate an array of a specific length filled with numbers incremented from start
def generate_array(length, start):
    return [start + i for i in range(length)]

# 3 - Fill an array, sort in ascending and descending order, and display output
def sort_array():
    arr = [int(input(f"Enter number {i+1}: ")) for i in range(5)]
    print("Ascending order:", sorted(arr))
    print("Descending order:", sorted(arr, reverse=True))

# 4 - FizzBuzz function
def fizz_buzz(n):
    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    else:
        return n

# 5 - Reverse a string
def reverse_string(s):
    return s[::-1]

# 6 - Calculate the area and circumference of a circle
def circle_properties():
    r = float(input("Enter the radius of the circle: "))
    area = 3.14159 * r * r
    circumference = 2 * 3.14159 * r
    print(f"Area: {area}")
    print(f"Circumference: {circumference}")

# 7 - Count occurrences of 'iti' in a string
def count_iti(s):
    return s.lower().count("iti")

# 8 - Find the longest alphabetical ordered substring
def longest_alphabetical_substring(s):
    longest = current = s[0]
    for i in range(1, len(s)):
        if s[i] >= s[i-1]:
            current += s[i]
        else:
            current = s[i]
        if len(current) > len(longest):
            longest = current
    return longest

# Example calls
print("Vowel count:", count_vowels("hello world"))
print("Generated array:", generate_array(5, 10))
sort_array()
print("FizzBuzz result:", fizz_buzz(15))
print("Reversed string:", reverse_string("hello"))
circle_properties()
print("'iti' count:", count_iti("itinerary iti initiative"))
print("Longest alphabetical substring:", longest_alphabetical_substring("abdulrahman"))
