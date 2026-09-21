// Prompt pool for the streams showcase. Six families, ~25 each. Rules: plain
// English, one request per prompt, an explicit length bound, no inner double
// quotes, no duplicates, nothing that invites a long answer or a refusal.
// Entries marked `bench` are verbatim from deploy/inkling-fleet/bench.py.

export const PROMPTS: readonly string[] = [
  // ---- Facts -------------------------------------------------------------
  "What is the capital of France? Answer in one word.", // bench
  "What is the boiling point of water? Explain.", // bench
  "Name a famous painting and its painter.", // bench
  "How many continents are there? Answer in one sentence.",
  "Which planet is closest to the Sun? One sentence.",
  "What is the largest mammal on Earth? One sentence.",
  "In which year did humans first land on the Moon? One sentence.",
  "What is the chemical symbol for gold? Answer in one word.",
  "Which language has the most native speakers? One sentence.",
  "What is the tallest mountain on Earth? One sentence.",
  "Who wrote Romeo and Juliet? Answer in a few words.",
  "What is the smallest prime number? One sentence.",
  "How many legs does a spider have? One sentence.",
  "What is the hardest natural substance? One sentence.",
  "Which ocean is the deepest? One sentence.",
  "What gas do plants absorb from the air? One sentence.",
  "What is the longest river in Africa? One sentence.",
  "How many bones are in the adult human body? One sentence.",
  "What is the freezing point of water in Fahrenheit? One sentence.",
  "Which metal is liquid at room temperature? One sentence.",
  "What is the currency of Japan? Answer in one word.",
  "Name the four seasons in order, starting with spring.",
  "What does DNA stand for? One sentence.",
  "Which instrument has 88 keys? One sentence.",
  "Roughly how fast is light in kilometres per second? One sentence.",

  // ---- Explain simply ---------------------------------------------------
  "Explain in three sentences why the sky is blue.", // bench
  "Why is the ocean salty? Two sentences.", // bench
  "Summarise photosynthesis in two sentences.", // bench
  "What does a compiler do? Two sentences.", // bench
  "Explain gravity to a child in two sentences.", // bench
  "Give one tip for sleeping better, in two sentences.", // bench
  "Explain how a rainbow forms, in two sentences.",
  "Explain what a black hole is to a ten-year-old, in two sentences.",
  "Why do we have leap years? Two sentences.",
  "Explain how a refrigerator keeps food cold, in two sentences.",
  "Why does ice float on water? Two sentences.",
  "Explain what the internet is to a ten-year-old, in two sentences.",
  "How does a battery store energy? Two sentences.",
  "Why do leaves change colour in autumn? Two sentences.",
  "Explain what inflation means, in two sentences.",
  "How do vaccines work? Two sentences, simple words.",
  "Why does bread rise? Two sentences.",
  "Explain what an algorithm is to a ten-year-old, in two sentences.",
  "Why is the sea blue but a glass of water clear? Two sentences.",
  "How do aeroplanes stay in the air? Two sentences.",
  "Explain what a mixture-of-experts model is, in two sentences.",
  "Why do cats purr? Two sentences.",
  "Explain what a GPU does, in two sentences.",
  "How does a thermostat work? Two sentences.",
  "Why does the Moon have phases? Two sentences.",
  "Explain what encryption is to a ten-year-old, in two sentences.",

  // ---- Short creative ----------------------------------------------------
  "Write two sentences about the Pacific Ocean.", // bench
  "Describe a cat in two sentences.", // bench
  "Describe rain in one sentence.", // bench
  "What is a haiku? Give one.", // bench
  "Write a haiku about a mountain at dawn.",
  "Write a haiku about a busy train station.",
  "Write a two-line rhyme about coffee.",
  "Write a one-sentence story about a lost umbrella.",
  "Describe a thunderstorm in two sentences.",
  "Write a haiku about autumn leaves.",
  "Describe a lighthouse at night in two sentences.",
  "Write a two-line rhyme about a sleepy dog.",
  "Write a one-sentence story about a robot learning to paint.",
  "Describe the smell of fresh bread in one sentence.",
  "Write a haiku about the first snow of winter.",
  "Describe a desert at noon in two sentences.",
  "Write a two-line rhyme about the sea.",
  "Write a one-sentence story about a key that opens nothing.",
  "Describe a city waking up in two sentences.",
  "Write a haiku about a river in summer.",
  "Describe an old library in two sentences.",
  "Write a two-line rhyme about the Moon.",
  "Write a one-sentence story about a letter that arrived fifty years late.",
  "Describe a forest after rain in two sentences.",
  "Write a haiku about a cup of tea.",

  // ---- Tiny code ---------------------------------------------------------
  "Write a three-line Python function that returns the square of a number.",
  "Write a one-line Python expression that reverses a string called s.",
  "Write a JavaScript function that checks whether a number is even, in three lines.",
  "Write a Python function that returns the larger of two numbers, in three lines.",
  "Write a SQL query that counts the rows in a table called users.",
  "Write a Bash one-liner that counts the lines in a file called log.txt.",
  "Write a Python list comprehension that squares the numbers 1 to 5.",
  "Write a Rust function that adds two i32 values, in three lines.",
  "Write a JavaScript arrow function that doubles a number, in one line.",
  "Write a Python function that returns True if a string is a palindrome, in three lines.",
  "Write a Go function that returns the length of a slice of ints, in three lines.",
  "Write a one-line Python expression that sums a list called nums.",
  "Write a CSS rule that centres the text in a class called title.",
  "Write a Python dictionary with three fruits as keys and their colours as values.",
  "Write a JavaScript function that returns the last element of an array, in one line.",
  "Write a SQL query that selects the name column from a table called cities, sorted alphabetically.",
  "Write a Python function that converts Celsius to Fahrenheit, in two lines.",
  "Write a Bash command that lists only the directories in the current folder.",
  "Write a TypeScript type for a point with x and y numbers.",
  "Write a Python function that returns the first n Fibonacci numbers, in five lines or fewer.",
  "Write a regular expression that matches a four-digit year.",
  "Write a one-line Python expression that checks whether 7 is in a list called xs.",
  "Write a C function that returns the maximum of two ints, in three lines.",
  "Write a JSON object describing a book with title, author and year.",
  "Write a Python function that counts the vowels in a string, in three lines.",

  // ---- Lists ---------------------------------------------------------------
  "Name two planets and one fact about each.", // bench
  "Name three primary colours, one word each.",
  "Name three programming languages, one word each.",
  "List three things you need to bake bread, one phrase each.",
  "Name three countries in South America, one word each.",
  "List three benefits of walking, one phrase each.",
  "Name three musical instruments with strings, one word each.",
  "List three uses for a paper clip, one phrase each.",
  "Name three famous scientists, one name each.",
  "List three things to pack for a beach day, one phrase each.",
  "Name three types of cloud, one word each.",
  "List three ways to save water at home, one phrase each.",
  "Name three board games, one name each.",
  "List three ingredients in a basic tomato sauce, one word each.",
  "Name three rivers in Europe, one name each.",
  "List three reasons to learn a second language, one phrase each.",
  "Name three kinds of renewable energy, one phrase each.",
  "List three things a good teacher does, one phrase each.",
  "Name three noble gases, one word each.",
  "List three tips for a good night of sleep, one phrase each.",
  "Name three shapes with four sides, one word each.",
  "List three things found in a kitchen drawer, one word each.",
  "Name three species of big cat, one word each.",
  "List three steps to make a cup of tea, one phrase each.",
  "Name three moons in the solar system, one word each.",

  // ---- Reasoning -----------------------------------------------------------
  "List three prime numbers and say why they are prime.", // bench
  "What is 6 times 7? Explain briefly.", // bench
  "If a train travels 60 km in one hour, how far does it go in two and a half hours? Show the calculation in one line.",
  "What is 15 percent of 200? Show one line of working.",
  "A dozen eggs cost 3 dollars. How much do 4 eggs cost? One sentence.",
  "Is 51 a prime number? Answer and explain in one sentence.",
  "What is the next number in the sequence 2, 4, 8, 16? One sentence.",
  "If today is Monday, what day is it in ten days? One sentence.",
  "What is 9 squared minus 1? One line of working.",
  "A rectangle is 4 by 6. What is its area and its perimeter? One sentence each.",
  "How many minutes are in a day? Show the calculation.",
  "Which is heavier, a kilogram of feathers or a kilogram of steel? One sentence.",
  "If you have 3 apples and eat one, how many are left? One sentence.",
  "What is half of a quarter? One sentence.",
  "Sort these numbers from smallest to largest: 17, 3, 42, 8. One line.",
  "What is 1000 divided by 8? One line of working.",
  "A clock shows 3:45. How many minutes until 4:30? One sentence.",
  "What is the sum of the first five odd numbers? Show the working in one line.",
  "If five pens cost 10 dollars, how much do eight pens cost? One sentence.",
  "Which is larger, 2 to the power of 10, or 1000? One sentence.",
  "How many sides do two hexagons have in total? One sentence.",
  "What is 12 times 12? Answer in one sentence.",
  "A car uses 5 litres per 100 km. How much fuel for 250 km? One line of working.",
  "Is the number 144 a perfect square? Answer and explain in one sentence.",
  "What is 7 times 8 plus 4? One line of working.",
];
