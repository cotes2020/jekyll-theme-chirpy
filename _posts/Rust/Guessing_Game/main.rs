extern crate rand;
use std::io;
use rand::Rng;
use std::cmp::Ordering;
fn main() {
   println!("Guessing game.....");
   let secret_number: u32  =  rand::thread_rng().gen_range(1,101);
   println!("The secret number is {}",secret_number);
   loop {
      let mut guess =  String::new();
      io::stdin().read_line(&mut guess)
         .expect("Enter a number.....!!!!");
      let guess: u32 = match guess.trim().parse() {
          Ok(num) => num,
          Err(_) => continue,
      };
      match guess.cmp(&secret_number) {
        Ordering::Less => println!("Lesser than the number"),
        Ordering::Equal => {
            println!("Equal to the number");
            break;
            },
        Ordering::Greater => println!("Greater than the number"),    
      }
   }
}
