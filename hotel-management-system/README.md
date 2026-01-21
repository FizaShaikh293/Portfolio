# Hotel Management System

A web-based Hotel Management System developed using PHP and MySQL to automate hotel room booking, reservation management, and administrative operations. This project was developed as part of an academic final year project and demonstrates a complete CRUD-based hotel booking workflow with separate user and admin modules.

## Features
User room reservation system  
Admin dashboard for booking confirmation  
Room availability management  
Payment and profit tracking  
Session-based authentication  
MySQL database integration  

## Technologies Used
PHP  
MySQL (XAMPP)  
HTML5  
CSS3  
Bootstrap 3  
JavaScript  
jQuery  

## Project Structure
hotel-management-system/  
├── admin/  
│   ├── home.php  
│   ├── roombook.php  
│   ├── payment.php  
│   ├── profit.php  
│   └── logout.php  
├── user/  
│   ├── index.php  
│   └── reservation.php  
├── config/  
│   └── db.php  
├── database/  
│   ├── hotel.sql  
│   └── payment.sql  
├── assets/  
│   ├── css/  
│   │   └── custom-styles.css  
│   ├── js/  
│   │   └── custom-scripts.js  
│   └── fonts/  
├── index.php  
├── .gitignore  
└── README.md  

## External Dependencies
This project uses external libraries that are not included in the repository.

Bootstrap 3  
Font Awesome  
jQuery  

You may either download these libraries manually or use CDN links.

Bootstrap 3: https://getbootstrap.com/docs/3.4/getting-started/  
jQuery: https://jquery.com/download/  
Font Awesome: https://fontawesome.com/v4.7.0/

## Database Setup
1. Install XAMPP  
2. Start Apache and MySQL  
3. Open phpMyAdmin  
4. Create a database named `hotel`  
5. Import the SQL files from the `database` folder  

## How to Run the Project
1. Clone the repository  
2. Move the project folder into `htdocs`  
3. Configure database credentials in `config/db.php`  
4. Import database SQL files  
5. Open browser and go to  
   http://localhost/hotel-management-system  

## Admin Access
Admin login is session-based and handled internally. After login, the admin can confirm bookings, view payments, and check total profit.

## Academic Context
This project was developed as part of a Bachelor of Science in Information Technology final year project and is intended for educational and demonstration purposes.

## Future Enhancements
Payment gateway integration  
Improved UI with modern Bootstrap version  
Role-based authentication  
Security enhancements using prepared statements  

## Author
Fiza Shaikh  

## License
This project is for academic and learning purposes only.
