import json
import random

def generate_mock_answers(filename="mock_answers.json"):
    choices = ["A", "B", "C", "D", "E"]
    mock_dict = {}
    
    # a) 25% din 200 = primele 50 intrebari (complement simplu)
    for i in range(1, 51):
        ans = [random.choice(choices)]
        
        # Hardcoding the explicitly checked ones from the image
        if i == 1: ans = ["A"]
        elif i == 2: ans = ["C"]
        elif i == 3: ans = ["B"] # Grila oficiala trebuie sa aiba 1 singur raspuns. Studentul a bifat doua ('B', 'D'), deci va fi anulat pe buna dreptate.
        elif i == 4: ans = ["A"] 
            
        mock_dict[str(i)] = ans

    # b) 75% din 200 = intrebarile 51-200 (complement multiplu)
    for i in range(51, 201):
        # 2-4 raspunsuri corecte
        count = random.randint(2, 4)
        ans = sorted(random.sample(choices, count))
        
        # Hardcoding the explicitly checked ones from the image
        if i == 102: ans = ["C", "E"] # Must be multiple stringency
        elif i == 103: ans = ["D", "A"]
            
        mock_dict[str(i)] = ans
        
    with open(filename, "w") as f:
        json.dump(mock_dict, f, indent=4)
        
    print(f"Generat cu succes 200 de intrebari in {filename}")

if __name__ == "__main__":
    generate_mock_answers()
