const express = require('express');
const router = express.Router();
const User = require('../models/user');  // Import User model
const path=require('path');
// POST /login route
router.post('/login', async (req, res) => {
    const { username, password } = req.body;
    console.log('📥 Received login attempt:', username, password);

    try {
        const user = await User.findOne({ where: { username, password } });
        console.log('🔍 User lookup result:', user);

        if (user) {
            req.session.user = { id: user.userid, username: user.username };
            console.log("✅ User authenticated:", req.session.user);
            res.status(200).json({ message: "✅ User authenticated!", user: req.session.user });
        } else {
            console.log("❌ Invalid credentials for:", username);
            res.status(401).json({ message: "❌ Invalid credentials!" });
        }
    } catch (error) {
        console.error("🔥 Server Error:", error);
        res.status(500).json({ message: "Internal server error" });
    }
});


// router.get('/addpatient', (req, res) => {
//     console.log("redirected to patient page");
//     res.sendFile(path.join(__dirname, '../public', 'addpatient.html'));
// });

router.get('/dashboard',(req,res)=>{
    console.log("redirected to dashboard");
    res.sendFile(path.join(__dirname,'../public','dashboard.html'));
})
module.exports = router;
