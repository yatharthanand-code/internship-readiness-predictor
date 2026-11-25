def main():
    print("Internship Readiness Predictor")
    print("Answer yes or no to each question.\n")

    questions = [
        "Do you have a resume?",
        "Have you completed a project related to your field?",
        "Can you describe your skills in an interview?",
        "Have you practiced interview questions?",
        "Do you have a LinkedIn profile or portfolio?",
    ]

    score = 0

    for question in questions:
        answer = input(f"{question} ").strip().lower()
        if answer in {"yes", "y"}:
            score += 1

    percentage = round(score / len(questions) * 100)

    print(f"\nReadiness score: {percentage}%")

    if percentage >= 80:
        print("You're looking internship-ready!")
    elif percentage >= 50:
        print("You're making progress. Keep building your skills and experience.")
    else:
        print("Start with a resume and a small project, then practice interviewing.")


if __name__ == "__main__":
    main()
