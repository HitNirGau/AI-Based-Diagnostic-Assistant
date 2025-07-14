const express = require('express');
const router = express.Router();
const path = require('path');

// Route for Landing Page
router.get('/patienthistory', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/patienthistory.html'));
});


router.get('/getadd',(req,res)=>{
    res.sendFile(path.join(__dirname, '../public/addpatient.html'));
})
router.get('/upload_xray' ,(req,res)=>{

    res.sendFile(path.join(__dirname,'../templates/upload_xray.html'));
})
module.exports = router;