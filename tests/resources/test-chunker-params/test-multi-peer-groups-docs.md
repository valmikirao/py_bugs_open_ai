### How ._chunk_by_peer_group looks in a CodeChunker instance for ./test-multi-peer-groups.py

```yaml

- - "class LargeClass1:\n    \n"

- - "def __init__(self):\n    \n"
  - "self.data = [x for x in range(1000000)]\n"

- - "def method1(self):\n    \n"
  - "result = 0\n"
  - "for i in range(len(self.data)):\n    result += self.data[i]\n"
  - "return result\n"

- - "def method2(self):\n    \n"
  - "result = 0\n"

- - "for i in range(len(self.data)):\n    \n"
  - "if self.data[i] % 2 == 0:\n    result += self.data[i]\n"

- - "return result\n"

- - "class LargeClass2:\n    \n"

- - "def __init__(self):\n    \n"
  - "self.data = [x**2 for x in range(1000000)]\n"

- - "def method1(self):\n    \n"
  - "result = 0\n"

- - "for i in range(len(self.data)):\n    \n"
  - "if self.data[i] < 500000000:\n    result += self.data[i]\n"

- - "return result\n"

- - "def method2(self):\n    \n"
  - "result = 0\n"

- - "for i in range(len(self.data)):\n    \n"
  - "if self.data[i] % 3 == 0:\n    result += self.data[i]\n"

- - "return result\n"
 ```