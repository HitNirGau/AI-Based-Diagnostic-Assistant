const express = require('express');
const router = express.Router();
const path = require('path');

// Route for Landing Page
router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public/landingpage.html'));
});

module.exports = router;