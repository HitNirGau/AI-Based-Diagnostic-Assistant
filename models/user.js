const { DataTypes } = require('sequelize');
const db = require('../config/db'); // Make sure this path points to your DB config file
const { Patient } = require('../models/addpatient');
const User = db.define('users', {
  userid: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true,
    allowNull: false,
  },
  username: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true,
  },
  password: {
    type: DataTypes.STRING,
    allowNull: false,
  },
}, {
  tableName: 'users',  // Your table name in PostgreSQL
  timestamps: false       // Disable createdAt and updatedAt fields
});

// User.hasMany(models.Patient, { foreignKey: 'userId', as: 'patients' });
User.associate = (models) => {  // Define associations within this function
  User.hasMany(models.Patient, { foreignKey: 'userId', as: 'patients' });
};

module.exports = User;
