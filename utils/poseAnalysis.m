function resultStruct = poseAnalysis(dlc_filename, likelihood_thres, outer_r_pixel, inner_r_pixel, arena_center,winA,winB,winC)
gaze_angle = 32.8; %degrees
% arena_center = [562, 437]; %pixels
if isa(arena_center,'cell')
    arena_center = cell2mat(arena_center);
    arena_center=cast(arena_center,'double');
end

if isa(winA,'cell')
    %winA = [cell2mat(winA{1});cell2mat(winA{2})];
    winA = cell2mat(winA);
    winA=cast(winA,'double');
end

if isa(winB,'cell')
    %winB = [cell2mat(winB{1});cell2mat(winB{2})];
    winB = cell2mat(winB);
    winB=cast(winB,'double');
end

if isa(winC,'cell')
    %winC = [cell2mat(winC{1});cell2mat(winC{2})];
    winC = cell2mat(winC);
    winC=cast(winC,'double');
end

%winA = rad2deg([atan2(winA(1,2)-arena_center(2),winA(1,1)-arena_center(1)), atan2(winA(2,2)-arena_center(2),winA(2,1)-arena_center(1))]);
%winA = [rad2deg(cart2pol(winA(1,1)-arena_center(1),winA(1,2)-arena_center(2))), rad2deg(cart2pol(winA(2,1)-arena_center(1),winA(2,2)-arena_center(2)))];
%winB = rad2deg([atan2(winB(1,2)-arena_center(2),winB(1,1)-arena_center(1)), atan2(winB(2,2)-arena_center(2),winB(2,1)-arena_center(1))]);
%winC= rad2deg([atan2(winC(1,2)-arena_center(2),winC(1,1)-arena_center(1)), atan2(winC(2,2)-arena_center(2),winC(2,1)-arena_center(1))]);

%% read data and get the number of frame
D = dlmread(dlc_filename, ',', 3, 1);
Nframes = size(D,1);

%% acquire the original data
snout_x = D(:,1);
snout_y = D(:,2);
snout_L = D(:,3);

earL_x = D(:,4);
earL_y = D(:,5);
earL_L = D(:,6);

earR_x = D(:,7);
earR_y = D(:,8);
earR_L = D(:,9);

tailbase_x = D(:,10);
tailbase_y = D(:,11);
tailbase_L = D(:,12);

%% pass the threshold
ok_ind = snout_L > likelihood_thres & ...
    earL_L > likelihood_thres & ...
    earR_L > likelihood_thres & ...
    tailbase_L > likelihood_thres;

Nframes_passed = sum(ok_ind);

msg = sprintf('%d of %d frames passed likelihood threshold', Nframes_passed, Nframes);
disp(msg);

%% data that passed the threshold
snout_x = snout_x(ok_ind);
snout_y = snout_y(ok_ind);
earL_x = earL_x(ok_ind);
earL_y = earL_y(ok_ind);
earR_x = earR_x(ok_ind);
earR_y = earR_y(ok_ind);
tailbase_x = tailbase_x(ok_ind);
tailbase_y = tailbase_y(ok_ind);

headbase_x = mean([earL_x, earR_x],2);
headbase_y = mean([earL_y, earR_y],2);

resultStruct.headbase_x = headbase_x;
resultStruct.headbase_y = headbase_y;

dx = snout_x-headbase_x;
dy = snout_y-headbase_y;

body_center = zeros(Nframes_passed,2);
for i=1:Nframes_passed
    body_center(i,:) = polygonCentroid([earR_x(i), earR_y(i); earL_x(i), earL_y(i); snout_x(i), snout_y(i); tailbase_x(i) tailbase_y(i)]);
end

dx = body_center(:,1) - arena_center(1);
dy = body_center(:,2) - arena_center(2);
resultStruct.bodyCenterAngle = 180+rad2deg(cart2pol(dx,dy));


%% get head triangle area
head_triangle_area = zeros(Nframes_passed,1);
for i=1:Nframes_passed
  head_triangle_area(i) = abs(triangleArea([earR_x(i), earR_y(i); earL_x(i), earL_y(i); snout_x(i), snout_y(i)]));
end
%% get body triangle area
body_triangle_area = zeros(Nframes_passed,1);
for i=1:Nframes_passed
  body_triangle_area(i) = abs(triangleArea([earR_x(i), earR_y(i); earL_x(i), earL_y(i); tailbase_x(i), tailbase_y(i)]));
end

%% get binocular view
theta_rad = cart2pol(dx,dy);

new_theta_L = deg2rad(rad2deg(theta_rad) + gaze_angle);
new_theta_R = deg2rad(rad2deg(theta_rad) - gaze_angle);

[dx_L, dy_L] = pol2cart(new_theta_L, 1);
[dx_R, dy_R] = pol2cart(new_theta_R, 1);

gazeLeft_lines = [headbase_x, headbase_y, dx_L, dy_L];
gazeRight_lines = [headbase_x, headbase_y, dx_R, dy_R];
gazeFront_lines = [headbase_x, headbase_y, dx, dy];

%center_to_snout = sqrt((snout_x - arena_center(1)).^2 + (snout_y - arena_center(2)).^2);

circ_inner = double([arena_center, inner_r_pixel]);
circ_outer = double([arena_center, outer_r_pixel]);

intersect_outer_left = intersectLineCircle(gazeLeft_lines, repmat(circ_outer, [Nframes_passed, 1]));
%intersect_inner_left = intersectLineCircle(gazeLeft_lines, repmat(circ_inner, [Nframes_passed, 1]));

intersect_outer_right = intersectLineCircle(gazeRight_lines, repmat(circ_outer, [Nframes_passed, 1]));
%intersect_inner_right = intersectLineCircle(gazeRight_lines, repmat(circ_inner, [Nframes_passed, 1]));

intersect_outer_front = intersectLineCircle(gazeFront_lines,repmat(circ_outer, [Nframes_passed, 1]));
%intersect_inner_front = intersectLineCircle(gazeFront_lines,repmat(circ_inner, [Nframes_passed, 1]));

outer_angles_left = zeros(Nframes_passed,1);
inner_angles_left = zeros(Nframes_passed,1);

outer_angles_right = zeros(Nframes_passed,1);
inner_angles_right = zeros(Nframes_passed,1);

outer_angles_front = zeros(Nframes_passed,1);
inner_angles_front = zeros(Nframes_passed,1);

%left eye
for i=1:Nframes_passed
   I_outer = intersect_outer_left(:,:,i);
   %I_inner = intersect_inner_left(:,:,i);
   
   %figure out correct outer intersection
   if isPointOnRay(I_outer(1,:), gazeLeft_lines(i,:))
      outer_pt = I_outer(1,:);
   else
      outer_pt = I_outer(2,:);
   end
   
   %figure out closest inner intersection
   I_inner = intersectLineCircle(gazeLeft_lines(i,:), circ_inner);
   if sum(isnan(I_inner),'all')
       %inner is not on the ray, so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_left(i) = rad2deg(cart2pol(dx,dy));      
       inner_angles_left(i) = nan;
   elseif ~isPointOnRay(I_inner(1,:), gazeLeft_lines(i,:)) %inner intersection on wrong side
       %so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_left(i) = rad2deg(cart2pol(dx,dy));    
       inner_angles_left(i) = nan;
   else %outer intersection occludded by inner
       L1 = edgeLength(gazeLeft_lines(i,1:2), I_inner(1,:));
       L2 = edgeLength(gazeLeft_lines(i,1:2), I_inner(2,:));
       if L1<L2
           dx = I_inner(1,1) - arena_center(1);
           dy = I_inner(1,2) - arena_center(2);
       else
           dx = I_inner(2,1) - arena_center(1); 
           dy = I_inner(2,2) - arena_center(2);
       end
       inner_angles_left(i) = rad2deg(cart2pol(dx,dy));  
       outer_angles_left(i) = nan;
   end

   dist_snout2center = sqrt((snout_x(i)-arena_center(1))^2+(snout_y(i)-arena_center(2))^2);
   logic_snout2center = dist_snout2center>outer_r_pixel | dist_snout2center<inner_r_pixel;

   dist_earL2center = sqrt((earL_x(i)-arena_center(1))^2+(earL_y(i)-arena_center(2))^2);
   logic_earL2center = dist_earL2center>outer_r_pixel | dist_earL2center<inner_r_pixel;

   dist_earR2center = sqrt((earR_x(i)-arena_center(1))^2+(earR_y(i)-arena_center(2))^2);
   logic_earR2center = dist_earR2center>outer_r_pixel | dist_earR2center<inner_r_pixel;

   dist_tail2center = sqrt((tailbase_x(i)-arena_center(1))^2+(tailbase_y(i)-arena_center(2))^2);
   logic_tail2center = dist_tail2center>outer_r_pixel | dist_tail2center<inner_r_pixel;

   if logic_earL2center|logic_earR2center|logic_snout2center|logic_tail2center
       inner_angles_left(i)=nan;
       outer_angles_left(i)=nan;
    end
       
end

%right eye
for i=1:Nframes_passed
   I_outer = intersect_outer_right(:,:,i);
   
   %figure out correct outer intersection
   if isPointOnRay(I_outer(1,:), gazeRight_lines(i,:))
      outer_pt = I_outer(1,:);
   else
      outer_pt = I_outer(2,:);
   end
   
   %figure out closest inner intersection
   I_inner = intersectLineCircle(gazeRight_lines(i,:), circ_inner);
   if sum(isnan(I_inner),'all')
       %inner is not on the ray, so outer is correct
       dx = outer_pt(1) - arena_center(1); 
       dy = outer_pt(2) - arena_center(2);
       outer_angles_right(i) = rad2deg(cart2pol(dx,dy));      
       inner_angles_right(i) = nan;
   elseif ~isPointOnRay(I_inner(1,:), gazeRight_lines(i,:)) %inner intersection on wrong side
       %so outer is correct
       dx = outer_pt(1) - arena_center(1); 
       dy = outer_pt(2) - arena_center(2);
       outer_angles_right(i) = rad2deg(cart2pol(dx,dy));    
       inner_angles_right(i) = nan;
   else %outer intersection occludded by inner
       L1 = edgeLength(gazeRight_lines(i,1:2), I_inner(1,:));
       L2 = edgeLength(gazeRight_lines(i,1:2), I_inner(2,:));
       if L1<L2
           dx = I_inner(1,1) - arena_center(1); 
           dy = I_inner(1,2) - arena_center(2);
       else
           dx = I_inner(2,1) - arena_center(1); 
           dy = I_inner(2,2) - arena_center(2);
       end
       inner_angles_right(i) = rad2deg(cart2pol(dx,dy));  
       outer_angles_right(i) = nan;
   end
    dist_snout2center = sqrt((snout_x(i)-arena_center(1))^2+(snout_y(i)-arena_center(2))^2);
   logic_snout2center = dist_snout2center>outer_r_pixel | dist_snout2center<inner_r_pixel;

   dist_earL2center = sqrt((earL_x(i)-arena_center(1))^2+(earL_y(i)-arena_center(2))^2);
   logic_earL2center = dist_earL2center>outer_r_pixel | dist_earL2center<inner_r_pixel;

   dist_earR2center = sqrt((earR_x(i)-arena_center(1))^2+(earR_y(i)-arena_center(2))^2);
   logic_earR2center = dist_earR2center>outer_r_pixel | dist_earR2center<inner_r_pixel;

   dist_tail2center = sqrt((tailbase_x(i)-arena_center(1))^2+(tailbase_y(i)-arena_center(2))^2);
   logic_tail2center = dist_tail2center>outer_r_pixel | dist_tail2center<inner_r_pixel;

   if logic_earL2center|logic_earR2center|logic_snout2center|logic_tail2center
       inner_angles_right(i)=nan;
       outer_angles_right(i)=nan;
   end
end

%front
for i=1:Nframes_passed
    I_outer = intersect_outer_front(:,:,i);

    if isPointOnRay(I_outer(1,:),gazeFront_lines(i,:))
        outer_pt=I_outer(1,:);
    else
        outer_pt = I_outer(2,:);
    end

    %figure out closest inner intersection
   I_inner = intersectLineCircle(gazeFront_lines(i,:), circ_inner);
   if sum(isnan(I_inner),'all')
       %inner is not on the ray, so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_front(i) = rad2deg(cart2pol(dx,dy));
       inner_angles_front(i) = nan;
   elseif ~isPointOnRay(I_inner(1,:), gazeFront_lines(i,:)) %inner intersection on wrong side
       %so outer is correct
       dx = outer_pt(1) - arena_center(1);
       dy = outer_pt(2) - arena_center(2);
       outer_angles_front(i) = rad2deg(cart2pol(dx,dy));
       inner_angles_front(i) = nan;
   else %outer intersection occludded by inner
       L1 = edgeLength(gazeFront_lines(i,1:2), I_inner(1,:));
       L2 = edgeLength(gazeFront_lines(i,1:2), I_inner(2,:));
       if L1<L2
           dx = I_inner(1,1) - arena_center(1);
           dy = I_inner(1,2) - arena_center(2);
       else
           dx = I_inner(2,1) - arena_center(1);
           dy = I_inner(2,2) - arena_center(2);
       end
       inner_angles_front(i) = rad2deg(cart2pol(dx,dy));
       outer_angles_front(i) = nan;
   end

   dist_snout2center = sqrt((snout_x(i)-arena_center(1))^2+(snout_y(i)-arena_center(2))^2);
   logic_snout2center = dist_snout2center>outer_r_pixel | dist_snout2center<inner_r_pixel;

   dist_earL2center = sqrt((earL_x(i)-arena_center(1))^2+(earL_y(i)-arena_center(2))^2);
   logic_earL2center = dist_earL2center>outer_r_pixel | dist_earL2center<inner_r_pixel;

   dist_earR2center = sqrt((earR_x(i)-arena_center(1))^2+(earR_y(i)-arena_center(2))^2);
   logic_earR2center = dist_earR2center>outer_r_pixel | dist_earR2center<inner_r_pixel;

   dist_tail2center = sqrt((tailbase_x(i)-arena_center(1))^2+(tailbase_y(i)-arena_center(2))^2);
   logic_tail2center = dist_tail2center>outer_r_pixel | dist_tail2center<inner_r_pixel;

   if logic_earL2center|logic_earR2center|logic_snout2center|logic_tail2center
       inner_angles_front(i)=nan;
       outer_angles_front(i)=nan;
   end

end


resultStruct.outer_angles_left = 360+outer_angles_left;
resultStruct.inner_angles_left = 360+inner_angles_left;
resultStruct.outer_angles_right = 360+outer_angles_right;
resultStruct.inner_angles_right = 360+inner_angles_right;
resultStruct.inner_angles_front = 360+inner_angles_front;
resultStruct.outer_angles_front = 360+outer_angles_front;
resultStruct.head_triangle_area = head_triangle_area;
resultStruct.body_triangle_area = body_triangle_area;

A_right = sum(resultStruct.outer_angles_right>=winA(1) & resultStruct.outer_angles_right<=winA(2));
B_right = sum(resultStruct.outer_angles_right>=winB(1) & resultStruct.outer_angles_right<=winB(2));
C_right = sum(resultStruct.outer_angles_right>=winC(1) & resultStruct.outer_angles_right<=winC(2));

A_left = sum(resultStruct.outer_angles_left>=winA(1) & resultStruct.outer_angles_left<=winA(2));
B_left = sum(resultStruct.outer_angles_left>=winB(1) & resultStruct.outer_angles_left<=winB(2));
C_left = sum(resultStruct.outer_angles_left>=winC(1) & resultStruct.outer_angles_left<=winC(2));

A_body = sum(resultStruct.bodyCenterAngle>=winA(1) & resultStruct.bodyCenterAngle<=winA(2));
B_body = sum(resultStruct.bodyCenterAngle>=winB(1) & resultStruct.bodyCenterAngle<=winB(2));
C_body = sum(resultStruct.bodyCenterAngle>=winC(1) & resultStruct.bodyCenterAngle<=winC(2));

resultStruct.A_weight_right = A_right ./  (A_right + B_right + C_right);
resultStruct.B_weight_right = B_right ./  (A_right + B_right + C_right);
resultStruct.C_weight_right = C_right ./  (A_right + B_right + C_right);

resultStruct.A_weight_left = A_left ./  (A_left+ B_left + C_left);
resultStruct.B_weight_left = B_left ./  (A_left + B_left + C_left);
resultStruct.C_weight_left = C_left ./  (A_left + B_left + C_left);

resultStruct.A_weight_body = A_body ./  (A_body+ B_body + C_body);
resultStruct.B_weight_body = B_body ./  (A_body + B_body + C_body);
resultStruct.C_weight_body = C_body ./  (A_body + B_body + C_body);

resultStruct.winPreference_right = ((A_right + B_right + C_right) / length(resultStruct.outer_angles_right)) * 360/99;
resultStruct.winPreference_left = ((A_left + B_left + C_left) / length(resultStruct.outer_angles_left)) * 360/99;
resultStruct.winPreference_body = ((A_body + B_body + C_body) / length(resultStruct.bodyCenterAngle)) * 360/99;





