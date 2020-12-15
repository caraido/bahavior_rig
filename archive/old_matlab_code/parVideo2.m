%todo: 
% consider making dedicated filewrite workers
%       would simplify acq code somewhat and possibly reduce droprate
%       more data sent by client? is this an issue?


nCameras = 2;
nWorkers = 4;

parpool(nWorkers);

acqQ = parallel.pool.PollableDataQueue;

%make queue on each worker back to the client that will persist (constant)
QC = parallel.pool.Constant(@parallel.pool.PollableDataQueue);

%clear acqF
%acqF(1:nCameras) = parallel.FevalFuture;

%each element in clientQ is a pollable queue used to send data to the workers
clientQ = fetchOutputs(parfevalOnAll(@(x) x.Value, 1, QC));


delete(imaqfind);
imaqreset;
parfevalOnAll(@eval,0,'delete(imaqfind)');
parfevalOnAll(@imaqreset,0);

%initialize each worker


F = parfevalOnAll(@parVideoWorker,1,acqQ,QC); 
pause(1); %allow workers to reach poll step
for i = 1:nWorkers
    %tell each worker its index number and role
    if i<=nCameras
        send(clientQ(i),{i,i,'acquisition'});
    else
        send(clientQ(i),{i,i-nCameras,'processing'});
    end
    %wait for each worker to finish initializing
    [msg,nohang] = poll(acqQ,10);
    if nohang
        if strcmp(msg{2},'Ready')
            fprintf('Worker %d initialized successfully.\n',i);
        else
            throw(msg{3});
        end
    else
        error('Something went wrong!');
    end
end


%when ready, start acquisition by sending message to the workers
for i = 1:nCameras
    send(clientQ(i),'beginAcq');
    %send(clientQ(i),'endAcq');
end

fps=15;
frameInterval=1/fps;
gpuBufferSize = 1024*1280*40; %make sure we send at least 100 frames at a time
currentBuffer=zeros(nCameras,1);
clf;im = imagesc(zeros(256,320,1)); caxis([0 1]); colormap('gray')
data = cell(nWorkers,1);
frameTime = cell(nWorkers,1);
%verbose = true;
verbose = false;


t=tic;lastt=[];
while toc(t)<1000
    %get data from workers
    if acqQ.QueueLength
        d=poll(acqQ);
        if numel(d)>=2
            switch class(d{2})
                case 'uint16' %raw image
                    workerInd = d{1};
                    d{2}=d{2}(:,:,:,1:5:end);
                    data{workerInd} = cat(4,data{workerInd},d{2});
                    frameTime{workerInd} = cat(1,frameTime{workerInd},d{3});
                    if verbose
                        fprintf('Collected %d frames from camera %d!\n',size(d{2},4),workerInd);
                    end
                    currentBuffer(workerInd) = numel(data{workerInd});
                case 'logical' %processed image
                    workerInd = d{1};
                    data{workerInd} = cat(3,data{workerInd},d{2});
                    %if verbose
                        fprintf('Collected %d frames from worker %d!\n',size(d{2},3),workerInd);
                    %end
                case 'char' %worker message
                    if verbose
                        fprintf('Message from worker %d: %s\n',d{1},d{2});
                    end
            end
        end
    end
    
    %send data to gpu workers
    %currently each gpu worker is tied to a camera
    %but maybe we ought to send out irrespective of camera
    % we can use ready signal from workers to decide which gpu to send to
    gpuSend = find(currentBuffer>=gpuBufferSize);
    if ~isempty(gpuSend)
        for n=gpuSend
            target = n+nCameras;
            send(clientQ(target),{'loadData',data{n}});
            %send(clientQ(target),data{n}); %workers don't seem to be getting data...
            if verbose
                fprintf('Sent data from camera %d to worker %d!\n',n,target);
            end
            data{n}=[];currentBuffer(n)=0;
        end 
    end
    
    %display processed frames
    if ~isempty(data{3})
        if isempty(lastt) || toc(lastt)>frameInterval %might want to record these tocs to get a sense of displayed framerate
            lastt=tic;
            set(im,'cdata',data{3}(1:4:end,1:4:end,1,1));
            data{3}(:,:,1)=[]; %pop the data from the buffer
            if verbose
                fprintf('Drawing frame\n');
            end
            drawnow;
        end
    end
end

for i = 1:nCameras
    send(clientQ(i),'pauseAcq')
end
pause(1); %let the worker close up...
cancel(F);
%     frame = d{2};
%     hi.CData = frame;
%     drawnow;
    
    
    
    %poll the queue to see if data is available
    %if new acquisition data
    %   send data to one of the processing workers
    %if newly processed data
    %   add to display queue
    %   if queue has grown too large, drop some older frames
    
    %if time to update display
    %   display next frame
    
    %if user cancels acquisition
    %   send message to each worker to cancel operation
%end











% for i=1:nCameras
%     acqF(i) = parfeval(@acquireCameraData,1,acqQ,acqQC);
%     %wait for signal from each worker that it has been connected to camera
%     %before starting next camera...
% end
% 
% 
% while
%     [data, isData] = poll(acqQ); %can specify a timeout period
%     if isData
%         
%     end
%     
% end

    
