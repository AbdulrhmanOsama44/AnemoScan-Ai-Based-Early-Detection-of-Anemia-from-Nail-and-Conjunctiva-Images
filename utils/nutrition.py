# Nutrition Guidance
def get_nutrition_advice(label):
    if label == "Anemic":
        return """
### 🥗 Nutrition Recommendations for Anemia

Iron-rich foods:
- Spinach, kale, broccoli
- Red meat, liver
- Lentils, chickpeas, beans

Vitamin C (helps iron absorption):
- Oranges, lemons
- Strawberries
- Bell peppers

Avoid with iron meals:
- Tea and coffee (reduce iron absorption)

Tip:
Combine iron + vitamin C (e.g., lentils + lemon juice)

This is NOT a medical diagnosis. Please consult a doctor. ❗️
"""
    else:
        return """
### ✅ Healthy Maintenance Tips

- Maintain a balanced diet
- Eat fruits and vegetables daily
- Stay hydrated
- Regular medical check-ups are recommended
"""