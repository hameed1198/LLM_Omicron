def is_palindrome_string(s):
    # Filter and normalize characters: only alphanumeric, all lowercase
    filtered = []
    for ch in s:
        if '0' <= ch <= '9' or 'a' <= ch.lower() <= 'z':
            filtered.append(ch.lower())
    # Two-pointer check
    left, right = 0, len(filtered) - 1
    while left < right:
        if filtered[left] != filtered[right]:
            return False
        left += 1
        right -= 1
    return True

# Example usage:
print(is_palindrome_string("A man, a plan, a canal: Panama"))  # True
print(is_palindrome_string("Hello"))                          # False


#reverse of string 

def reverse_of_string(s):
    # Filter and normalize characters: only alphanumeric, all lowercase
    filtered = []
    for ch in s:
        if '0' <= ch <= '9' or 'a' <= ch.lower() <= 'z':
            filtered.append(ch.lower())
    # Reverse the filtered string
    return ''.join(reversed(filtered))

print(reverse_of_string('Khan'))



#

def duplicate_string(s):
    # Filter and normalize characters: only alphanumeric, all lowercase
    filtered = []
    for ch in s:
        if '0' <= ch <= '9' or 'a' <= ch.lower() <= 'z':
            filtered.append(ch.lower())
    # Duplicate the filtered string
    return ''.join(filtered) * 2

print(duplicate_string('Khan'))  # Output: "khankhan"

def identify_duplicates_list(lst):
    # Create a set to track seen elements
    seen = set()
    duplicates = set()
    
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)

print(identify_duplicates_list([1, "kajskds", 3, 4, "kajskds", 1, 2]))  # Output: [1, 2]

def identify_child_class(cls):
    # Get all subclasses of the given class
    subclasses = cls.__subclasses__()
    
    # Filter out the subclasses that are not child classes
    child_classes = [subclass for subclass in subclasses if subclass.__name__ != 'ChildClass']
    
    return child_classes

class ParentClass:
    pass
class ChildClass(ParentClass):
    pass
class AnotherChildClass(ParentClass):
    pass
print(identify_child_class(ParentClass))  # Output: [<class '__main__.ChildClass'>, <class '__main__.AnotherChildClass'>]


def identify_child_node(node):
    # Get all child nodes of the given node
    child_nodes = node.get_children()
    
    # Filter out the nodes that are not child nodes
    child_nodes = [child for child in child_nodes if child.is_child()]
    
    return child_nodes

class Node:
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
    
    def get_children(self):
        return self.children
    
    def is_child(self):
        return True  # For simplicity, assume all nodes are child nodes
    #parent node

class ParentNode(Node):
    def is_child(self):
        return False  # Parent nodes are not child nodes
    
# Example usage
root = Node("Root")
child1 = Node("Child1")
child2 = Node("Child2")
root.add_child(child1)
root.add_child(child2)
print(identify_child_node(root))  # Output: [<__main__.Node object at ...>, <__main__.Node object at ...>] 



# tower of hanoi
def tower_of_hanoi(n, source, target, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n - 1, source, auxiliary, target)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n - 1, auxiliary, target, source)
# Example usage
n = 3  # Number of disks
tower_of_hanoi(n, 'A', 'C', 'B')  # A, B, C are the names of the rods

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
# Example usage
print(factorial(5))  # Output: 120

def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
# Example usage
print(fibonacci(10))  # Output: 5


