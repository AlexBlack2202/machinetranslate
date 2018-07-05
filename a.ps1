param (
    $remotePath = "/",
    $wildcard = "*.jpg"
)
 
try {
    # Load WinSCP .NET assembly
    Add-Type -Path "C:\Program Files (x86)\WinSCP\WinSCPnet.dll"
 
    # Setup session options
    $sessionOptions = New-Object WinSCP.SessionOptions -Property @{
        Protocol = [WinSCP.Protocol]::ftp
        HostName = "103.48.77.131"
        UserName = "pdtung"
        Password = "pdtung!@"
        #SshHostKeyFingerprint = "ssh-rsa 2048 xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
    }
	
    #$sessionOptions.GiveUpSecurityAndAcceptAnySshHostKey = $true
 
    $session = New-Object WinSCP.Session
	
	
 
    try {
        # Connect
        $session.Open($sessionOptions)
		
        # get all store 
		
		$remove_w = ".",".."
		
        $directory_store = $session.ListDirectory("/")
		
		$filedate = (Get-Date).Adddays(-2) 
		
        foreach ($storeInfo in $directory_store.Files) {
            if ($storeInfo.IsDirectory -And $remove_w -notcontains $storeInfo.Name) {
                $directory_store_ip = $session.ListDirectory("/$($storeInfo.Name)")
			
                foreach ($storeIP in $directory_store_ip.Files) {
                    if ($storeIP.IsDirectory -And $remove_w -notcontains $storeIP.Name) {
                        $directory_store_ip_day = $session.ListDirectory("/$($storeInfo.Name)/$($storeIP.Name)")
					
                        foreach ($storeDay in $directory_store_ip_day.Files) {
                            if ($storeDay.IsDirectory -And $remove_w -notcontains $storeDay.Name -And $storeDay.LastWriteTime -lt $filedate) {
							
                                $filename_b = "/"+$storeInfo.Name+"/"+$storeIP.Name+ "/" + $storeDay.Name

                                $removalResult = $session.RemoveFiles($filename_b)
 
                                if ($removalResult.IsSuccess)
                                {
                                    Write-Host "Removing of file 1 $($filename_b) succeeded"
                                }
                                else
                                {
                                    Write-Host "Removing of file $($filename_b) failed"
									 Write-Host "$($removalResult.Failures[0])"
                                }


                                # Write-Host ("$($storeDay.Name) with size $($storeDay.Length), " +
                                #     "permissions $($storeDay.FilePermissions) and " +
                                #     "last modification at $($storeDay.LastWriteTime)")
                            }
                        }
                    }
			
                }
            }
        }
    
    }
 

    finally {
        # Disconnect, clean up
        $session.Dispose()
    }
 
    exit 0
}
catch {
    Write-Host "Error: $($_.Exception.Message)"
    exit 1
}