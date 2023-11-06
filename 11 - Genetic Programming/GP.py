
class GP_Node:
	def __init__(self, label:str=None) -> None:
		self.label = label
	def evaluate(self) -> float:
		pass

class Terminal_N(GP_Node):
	def __init__(self, value, label:str=None) -> None:
		super().__init__(label)
		self.value = value
	def evaluate(self) -> float:
		return self.value
	def __str__(self) -> str:
		if self.label is None:
			return str(self.value)
		return self.label

class Function_N(GP_Node):
	def __init__(self,
		function,
		leftChild :GP_Node=None,
		rightChild:GP_Node=None,
		label:str=None
	) -> None:
		if label is None:
			label = str(function)
		super().__init__(label)
		self.func  = function
		self.leftChild  = leftChild
		self.rightChild = rightChild
	def evaluate(self) -> float:
		if self.leftChild is None or self.rightChild is None:
			raise ValueError("Null child error!")
		
		resLeft  = self.leftChild.evaluate()
		resRight = self.rightChild.evaluate()

		return self.func(resLeft, resRight)
	def __str__(self) -> str:
		return f'{str(self.leftChild)},{str(self.rightChild)}{str(self.label)}'
