import random

def computerChoice():
    
    computerChoice = random.randint(1,3)
    
    return computerChoice

def result(userInput, computerChoice):

    if userInput == computerChoice:
        return('tie')
    elif (userInput == 3 and computerChoice == 1) or userInput - 1 == computerChoice:
        return('You win!')
    else:
        return('You lose!')

userInput = input('Please input your choice \n')

computerChoice = computerChoice()

print(computerChoice)
print(result(int(userInput), computerChoice))