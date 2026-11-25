# people.py - word lists used to recognise person-to-person UPI payments (see categorizer.person_transfer).
# Written from general knowledge of common Indian first names; nothing here comes from any test data.

# Common first names (lowercase). Ambiguous words that are also everyday words were left out on purpose.
FIRST_NAMES = {
    # men
    "aarav", "aditya", "ajay", "akash", "akhil", "amit", "amar", "anand", "anil", "ankit", "anuj", "arjun",
    "arun", "ashish", "ashok", "bharat", "chetan", "deepak", "dinesh", "farhan", "gaurav", "gopal", "harsh",
    "harish", "hemant", "imran", "jatin", "karan", "kartik", "kunal", "lokesh", "mahesh", "manish", "mohan",
    "mukesh", "naveen", "neeraj", "nikhil", "nitin", "pankaj", "prakash", "pranav", "prashant", "pratik",
    "rahul", "raj", "rajesh", "rajiv", "rakesh", "ramesh", "ravi", "rohan", "rohit", "sachin", "sandeep",
    "sanjay", "santosh", "saurabh", "shubham", "siddharth", "srinivas", "suresh", "sumit", "sunil", "suraj",
    "tarun", "uday", "varun", "vijay", "vikas", "vikram", "vinay", "vinod", "vishal", "vivek", "yash",
    "yogesh", "zaid", "mohammed", "salman", "aman", "abhishek", "abhinav", "akshay", "ayush", "kapil",
    "manoj", "mayank", "naresh", "parth", "pavan", "praveen", "ritesh", "tushar", "venkat", "subhash",
    "jagdish", "vishnu", "girish", "ganesh", "mahendra", "dhruv", "ishaan", "kabir", "lalit", "nilesh",
    # women
    "aarti", "aditi", "akanksha", "ananya", "anita", "anjali", "ankita", "archana", "asha", "bhavna",
    "chitra", "deepa", "deepika", "disha", "divya", "geeta", "gayatri", "isha", "kajal", "kavita", "kavya",
    "komal", "kriti", "lakshmi", "lata", "madhuri", "meena", "megha", "nandini", "neha", "nidhi", "nisha",
    "pallavi", "pooja", "prachi", "pragya", "preeti", "priya", "priyanka", "rachna", "radha", "rekha",
    "renu", "riya", "ritu", "sakshi", "sangeeta", "sapna", "sarita", "shweta", "simran", "smita", "sneha",
    "sonal", "sonia", "sunita", "supriya", "suman", "swati", "tanvi", "vandana", "varsha", "vidya",
    "yamini", "zoya", "fatima", "ayesha", "shruti", "tanya", "trisha", "payal", "poonam", "reena",
}

# Family words people type instead of a name.
RELATIONS = {"mom", "dad", "mummy", "papa", "bhai", "bhaiya", "didi", "amma", "appa", "brother", "sister"}

# If one of these words is in the text it is a business (Rahul Medical Store), not a person.
BUSINESS_WORDS = {
    "store", "stores", "mart", "traders", "trader", "trading", "enterprises", "enterprise", "pvt", "ltd",
    "limited", "llp", "services", "service", "shop", "shoppe", "sweets", "sweet", "restaurant", "hotel",
    "medical", "medicals", "pharmacy", "pharma", "clinic", "hospital", "cafe", "bakery", "foods", "food",
    "retail", "supermarket", "electronics", "salon", "studio", "boutique", "company", "co", "corp",
    "bazaar", "kirana", "general", "agency", "agencies", "fashion", "fashions", "textiles", "garments",
    "jewellers", "jewellery", "hardware", "dairy", "tiffin", "bhandar", "emporium", "super", "market",
    "centre", "center", "mobile", "mobiles", "travels", "tours", "motors", "petrol", "fuel", "gas",
    "books", "opticals", "optical", "fitness", "gym", "school", "college", "academy", "institute",
    "classes", "tuition", "recharge", "bill", "fee", "fees", "charges", "interest", "atm", "pos", "ecom", "sales",
}
