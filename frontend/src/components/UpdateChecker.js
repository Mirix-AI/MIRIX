import React, { useEffect, useState } from 'react';
import UpdateModal from './UpdateModal';

const UpdateChecker = ({ currentVersion = '0.1.3' }) => {
  const [updateInfo, setUpdateInfo] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [checkingForUpdates, setCheckingForUpdates] = useState(false);

  const checkForUpdates = async () => {
    try {
      setCheckingForUpdates(true);
      console.log('Checking for updates...');
      
      const response = await fetch('https://api.github.com/repos/Mirix-AI/MIRIX/tags');
      if (!response.ok) {
        throw new Error('Failed to fetch tags');
      }
      
      const tags = await response.json();
      if (tags.length === 0) {
        console.log('No tags found');
        return;
      }
      
      const latestTag = tags[0].name;
      const latestVersion = latestTag.replace(/^v/, '');
      
      const isNewer = compareVersions(latestVersion, currentVersion) > 0;
      
      if (isNewer) {
        const releaseResponse = await fetch(`https://api.github.com/repos/Mirix-AI/MIRIX/releases/tags/${latestTag}`);
        let releaseInfo = null;
        
        if (releaseResponse.ok) {
          releaseInfo = await releaseResponse.json();
        }
        
        setUpdateInfo({
          version: latestVersion,
          tagName: latestTag,
          releaseNotes: releaseInfo?.body || 'No release notes available',
          releaseDate: releaseInfo?.published_at || tags[0].commit?.date,
          downloadUrl: releaseInfo?.html_url || `https://github.com/Mirix-AI/MIRIX/releases/tag/${latestTag}`
        });
        setIsModalOpen(true);
      } else {
        console.log('No updates available');
      }
    } catch (error) {
      console.error('Error checking for updates:', error);
    } finally {
      setCheckingForUpdates(false);
    }
  };

  const compareVersions = (v1, v2) => {
    const parts1 = v1.split('.').map(Number);
    const parts2 = v2.split('.').map(Number);
    
    for (let i = 0; i < Math.max(parts1.length, parts2.length); i++) {
      const part1 = parts1[i] || 0;
      const part2 = parts2[i] || 0;
      
      if (part1 > part2) return 1;
      if (part1 < part2) return -1;
    }
    
    return 0;
  };

  const handleUpdate = async () => {
    if (!updateInfo) return;
    
    // Simply open the download page and let user handle installation
    window.open(updateInfo.downloadUrl, '_blank');
  };

  useEffect(() => {
    // Check for updates 3 seconds after app starts (to not slow down initial load)
    const initialCheck = setTimeout(() => {
      checkForUpdates();
    }, 3000);
    
    // Then check every 12 hours
    const interval = setInterval(() => {
      checkForUpdates();
    }, 12 * 60 * 60 * 1000); // 12 hours
    
    return () => {
      clearTimeout(initialCheck);
      clearInterval(interval);
    };
  }, []);

  return (
    <UpdateModal
      isOpen={isModalOpen}
      onClose={() => setIsModalOpen(false)}
      updateInfo={updateInfo}
      currentVersion={currentVersion}
      onUpdate={handleUpdate}
    />
  );
};

export default UpdateChecker;