const express = require('express');
const session = require('express-session');
const bodyParser = require('body-parser');
const userRoutes = require('./routes/user');
const addPatientRoutes=require('./routes/Addpatient');  // Import routes
const path=require('path');
const mainRoutes = require('./routes/landingpage');
const patientHistoryRoutes = require('./routes/patienthistory');
const dashboardroutes=require('./routes/dashboard');
const app = express();
const PORT = 5001;

// Middleware
app.use(bodyParser.json());
app.use(session({
    secret: 'secret_key',
    resave: false,
    saveUninitialized: true,
}));

app.use(express.static('public'));
app.use('/data', express.static('data'));
app.use(express.static(path.join(__dirname, 'public')));


app.use('/landing',mainRoutes);
app.use('/', userRoutes); 
app.use('/dash',dashboardroutes); 
app.use('/history', patientHistoryRoutes); 

app.use('/add',addPatientRoutes);
// app.use('/history', require('./routes/patienthistory'));  // Assuming the routes are in 'routes/history.js'
// app.get('*', (req, res) => {
//     res.send('404 Not Found');
// });

app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
});
