use std::io;

fn main() {
   loop {
        let mut fahrenheit_degrees = String::new();
        println!("Enter the fahrenheit_degree: ");
        io::stdin().read_line(&mut fahrenheit_degrees)
            .expect("No value provided!!!!");
       let fahrenheit_degrees: i32 = match fahrenheit_degrees.trim().parse() {
           Ok(num) => num,
          Err(num) => continue,
       };
       println!("The value is {}\n",fahrenheit_tocelsius(fahrenheit_degrees));
   }
}
fn fahrenheit_tocelsius(fahrenheit_degrees: i32) -> i32{
   let mut calculation: i32 = (fahrenheit_degrees - 32) * 5;
   calculation = calculation/9;
   return calculation
}
