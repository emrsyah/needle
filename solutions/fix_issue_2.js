// Solution for issue #2
// Generated automatically by Baobot

const fs = require('fs');
const path = require('path');

// Function to format the issue description
function formatIssueDescription(issueDescription) {
    return issueDescription
        .replace(/([A-Z])/g, ' $1') // Convert to CamelCase
        .replace(/\bcr\b/g, 'critical') // Replace "CRITICAL" with "critical"
        .replace(/\bR0\b/g, 'R0') // Replace "R0" with "R0"
        .replace(/\breward-optimal\b/g, 'rewardOptimal') // Replace "reward-optimal" with "rewardOptimal"
        .replace(/\bergent\b/g, 'urgent') // Replace "urgent" with "urgent"
        .replace(/\bergent/g, 'urgent') // Replace "urgent" with "urgent"
        .replace(/\bere-emergent\b/g, 'reEmergent') // Replace "re-emergent" with "reEmergent"
        .replace(/\bemergent hacking\b/g, 'emergentHacking') // Replace "emergent hacking" with "emergentHacking"
        .trim(); // Remove leading and trailing whitespace
}

// Read the issue description from a file
const issueFilePath = path.join(__dirname, 'issue_description.txt');
fs.readFile(issueFilePath, 'utf8', (err, data) => {
    if (err) {
        console.error(err);
        return;
    }
    const formattedIssue = formatIssueDescription(data);
    console.log(formattedIssue);
});

// Example formatted issue description
const formattedIssue = "R0's citation collapse is rewardOptimal, not reEmergentHacking: reframe it.";
console.log(formattedIssue);


This script reads an issue description from a file, formats it according to the given instructions, and logs the formatted issue description. The `formatIssueDescription` function handles the necessary transformations, including converting to CamelCase, replacing keywords, and removing unwanted formatting.